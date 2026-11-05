"""Sample a balanced set of identity embeddings from a trained ID-Flow model.

The label space is every combination of age x race x gender
(9 x 7 x 2 = 126 groups). For each group we sample embeddings and greedily keep those whose
cosine similarity to every already accepted embedding of the group is below a threshold
(unit vectors: cosine similarity = dot product), until ``ceil(n_total / 126)`` identities
per group are collected.

    python scripts/sample_identities.py --ckpt idflow.pt --n_total 10000 \
        --output_dir outputs/identities

Outputs (in ``--output_dir``):
    id_embeddings.pt   FloatTensor (N, 512), L2-normalised, ordered by group
    labels.pkl         list of [age, race, gender] class indices, one entry per embedding
    metadata.json      run settings and per-group statistics

``--resume_from`` tops up an existing output directory (groups below target are sampled
further, optionally with a looser ``--relaxed_threshold``).
"""
import argparse
import json
import math
import os
import pickle
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from tqdm import tqdm

from idflow.io import load_idflow
from idflow.labels import (
    AGE_LABELS, GENDER_LABELS, N_GROUPS, RACE_LABELS, all_groups,
)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def sample_batch(model, group, bs, device, guidance_scale):
    """Sample ``bs`` L2-normalised embeddings (numpy, (bs, 512)) for one (age, race, gender) group."""
    x_init = torch.randn((bs, 512), device=device)
    x_init = x_init / torch.norm(x_init, dim=-1, keepdim=True)
    y = torch.tensor(group, dtype=torch.long, device=device).unsqueeze(0).expand(bs, -1)
    embs = model.sample(bs=bs, y=y, guidance_scale=guidance_scale, x_init=x_init)
    embs = embs / torch.norm(embs, dim=-1, keepdim=True)
    return embs.cpu().float().numpy()


def greedy_diversity_filter(candidates, accepted, threshold, target):
    """Accept candidates whose max cosine similarity to all accepted embeddings is < threshold."""
    for cand in candidates:
        if len(accepted) >= target:
            break
        if len(accepted) == 0:
            accepted = cand[None].copy()
        elif float((accepted @ cand).max()) < threshold:
            accepted = np.concatenate([accepted, cand[None]], axis=0)
    return accepted


def sample_group(model, group, accepted, n_per_group, bs, device, guidance_scale, threshold, max_iters):
    iters = 0
    for _ in range(max_iters):
        if len(accepted) >= n_per_group:
            break
        candidates = sample_batch(model, group, bs, device, guidance_scale)
        accepted = greedy_diversity_filter(candidates, accepted, threshold, n_per_group)
        iters += 1
    return accepted[:n_per_group], iters


def describe(group):
    a, r, g = group
    return f"{AGE_LABELS[a]:5s} | {RACE_LABELS[r]:15s} | {GENDER_LABELS[g]:6s}"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True, help="exported weights (.pt) or Lightning checkpoint (.ckpt)")
    p.add_argument("--model_config", default=None, help="training config; only needed for checkpoints without stored hyper-parameters")
    p.add_argument("--n_total", type=int, default=10_000, help="target number of identities")
    p.add_argument("--output_dir", default="outputs/identities")
    p.add_argument("--resume_from", default=None, help="existing output dir to top up")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--id_gs", type=float, default=2.0, help="identity guidance scale")
    p.add_argument("--similarity_threshold", type=float, default=0.3, help="max pairwise cosine similarity within a group")
    p.add_argument("--relaxed_threshold", type=float, default=None, help="looser threshold for groups that are short in --resume_from mode (e.g. 0.45)")
    p.add_argument("--bs", type=int, default=200, help="embeddings sampled per batch per group")
    p.add_argument("--max_iters", type=int, default=30, help="max sampling rounds per group")
    args = p.parse_args()

    device = torch.device(args.device)
    seed_everything(args.seed)
    n_per_group = math.ceil(args.n_total / N_GROUPS)
    print(f"{N_GROUPS} groups x {n_per_group} identities = {n_per_group * N_GROUPS} (>= {args.n_total})")

    model = load_idflow(args.ckpt, config_path=args.model_config, device=device)
    groups = all_groups()
    key = lambda g: f"{g[0]}_{g[1]}_{g[2]}"

    accepted = {key(g): np.empty((0, 512), dtype=np.float32) for g in groups}
    stats = {}
    threshold = args.similarity_threshold
    todo = groups
    if args.resume_from is not None:
        emb = torch.load(os.path.join(args.resume_from, "id_embeddings.pt")).numpy()
        with open(os.path.join(args.resume_from, "labels.pkl"), "rb") as f:
            labels = pickle.load(f)
        with open(os.path.join(args.resume_from, "metadata.json")) as f:
            stats = json.load(f).get("group_stats", {})
        for i, lbl in enumerate(labels):
            k = key(lbl)
            accepted[k] = np.concatenate([accepted[k], emb[i][None]], axis=0)
        todo = [g for g in groups if len(accepted[key(g)]) < n_per_group]
        print(f"Resuming: {len(todo)} / {N_GROUPS} groups below target")
        if args.relaxed_threshold is not None:
            threshold = args.relaxed_threshold

    os.makedirs(args.output_dir, exist_ok=True)
    for g in tqdm(todo, desc="Groups"):
        k = key(g)
        before = len(accepted[k])
        accepted[k], iters = sample_group(
            model, g, accepted[k], n_per_group, args.bs, device, args.id_gs, threshold, args.max_iters
        )
        if len(accepted[k]) < n_per_group:
            tqdm.write(f"[WARN] {describe(g)}: {before} -> {len(accepted[k])}/{n_per_group} after {iters} iterations")
        stats[k] = {
            "count": len(accepted[k]),
            "iters": stats.get(k, {}).get("iters", 0) + iters,
            "age": AGE_LABELS[g[0]], "race": RACE_LABELS[g[1]], "gender": GENDER_LABELS[g[2]],
        }

    all_emb = np.concatenate([accepted[key(g)] for g in groups], axis=0)
    all_labels = [list(g) for g in groups for _ in range(len(accepted[key(g)]))]

    torch.save(torch.from_numpy(all_emb), os.path.join(args.output_dir, "id_embeddings.pt"))
    with open(os.path.join(args.output_dir, "labels.pkl"), "wb") as f:
        pickle.dump(all_labels, f)
    meta = {
        "n_total": len(all_emb), "n_per_group": n_per_group, "n_groups": N_GROUPS,
        "similarity_threshold": args.similarity_threshold, "relaxed_threshold": args.relaxed_threshold,
        "id_gs": args.id_gs, "seed": args.seed, "bs": args.bs, "ckpt": os.path.basename(args.ckpt),
        "label_order": ["age", "race", "gender"],
        "age_labels": AGE_LABELS, "race_labels": RACE_LABELS, "gender_labels": GENDER_LABELS,
        "group_stats": stats,
    }
    with open(os.path.join(args.output_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    counts = [stats[key(g)]["count"] for g in groups if key(g) in stats]
    print(f"\nSaved {len(all_emb)} embeddings to {args.output_dir}")
    print(f"Per-group count range: [{min(counts)}, {max(counts)}] (target {n_per_group})")
    short = sum(c < n_per_group for c in counts)
    print(f"{short} groups below target" if short else "All groups reached the target")


if __name__ == "__main__":
    main()
