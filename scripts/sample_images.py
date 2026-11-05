"""Generate face images with Arc2Face from sampled identity embeddings.

For every identity embedding, up to ``--images_per_identity`` diverse images are generated.
Diversity is enforced by re-extracting ArcFace embeddings from the generated images and
greedily accepting an image only if its cosine similarity to all previously accepted images
of the same identity is below ``--similarity_threshold``. Images without a detectable face
are discarded.

Sampling can be split into parts (e.g. one per GPU) and merged afterwards:

    # part 0 of 4
    python scripts/sample_images.py sample \
        --embeddings_path outputs/identities/id_embeddings.pt \
        --output_dir outputs/images --part_idx 0 --n_parts 4

    # merge all parts (symlinks by default, --copy to duplicate files)
    python scripts/sample_images.py merge --output_dir outputs/images --n_parts 4

Progress is checkpointed after every identity, so interrupted runs resume automatically.
"""
import argparse
import json
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from idflow.arc2face_utils import build_arc2face_pipeline, build_face_analyzer, project_face_embs


def extract_arcface_embedding(face_analyzer, pil_image):
    """L2-normalised (512,) ArcFace embedding of the first detected face, or None."""
    img_bgr = cv2.cvtColor(np.array(pil_image.convert("RGB")), cv2.COLOR_RGB2BGR)
    faces = face_analyzer.get(img_bgr)
    if not faces:
        return None
    return faces[0].normed_embedding.astype(np.float32)


def sample_diverse_images(pipeline, face_analyzer, prompt_embeds, target, batch_size, steps,
                          guidance_scale, similarity_threshold, max_iters):
    """Returns (accepted PIL images, n generated, n without a detected face)."""
    accepted_images, accepted_embs = [], np.empty((0, 512), dtype=np.float32)
    n_generated = n_no_face = 0
    for _ in range(max_iters):
        if len(accepted_images) >= target:
            break
        result = pipeline(
            prompt_embeds=prompt_embeds, num_images_per_prompt=batch_size,
            num_inference_steps=steps, guidance_scale=guidance_scale,
        )
        n_generated += batch_size
        for img in result.images:
            if len(accepted_images) >= target:
                break
            emb = extract_arcface_embedding(face_analyzer, img)
            if emb is None:
                n_no_face += 1
            elif len(accepted_embs) == 0 or float((accepted_embs @ emb).max()) < similarity_threshold:
                accepted_images.append(img)
                accepted_embs = np.concatenate([accepted_embs, emb[None]], axis=0)
    return accepted_images, n_generated, n_no_face


def write_part_metadata(meta_path, args, start_idx, end_idx, stats):
    meta = {
        "part_idx": args.part_idx, "n_parts": args.n_parts,
        "start_idx": start_idx, "end_idx": end_idx,
        "images_per_identity": args.images_per_identity,
        "similarity_threshold": args.similarity_threshold,
        "steps": args.steps, "guidance_scale": args.guidance_scale,
        "stats": {str(k): v for k, v in stats.items()},
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)


def cmd_sample(args):
    device = args.device
    dtype = torch.float16 if "cuda" in device else torch.float32
    device_id = int(device.split(":")[-1]) if ":" in device else 0

    embeddings = torch.load(args.embeddings_path, map_location="cpu")
    if embeddings.dim() != 2 or embeddings.shape[1] != 512:
        raise ValueError(f"Expected an (N, 512) tensor, got {tuple(embeddings.shape)}")
    N = embeddings.shape[0]
    part_size = (N + args.n_parts - 1) // args.n_parts
    start_idx = args.part_idx * part_size
    end_idx = min(start_idx + part_size, N)
    print(f"{N} identities; part {args.part_idx}/{args.n_parts} covers [{start_idx}, {end_idx})")

    part_dir = os.path.join(args.output_dir, f"part_{args.part_idx:04d}")
    images_dir = os.path.join(part_dir, "images")
    os.makedirs(images_dir, exist_ok=True)
    meta_path = os.path.join(part_dir, "metadata.json")

    stats = {}
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            stats = {int(k): v for k, v in json.load(f).get("stats", {}).items()}
        print(f"Resuming: {len(stats)} identities already done")

    pipeline = build_arc2face_pipeline(args.arc2face_models_dir, device, dtype)
    face_analyzer = build_face_analyzer(args.insightface_root, device_id)

    for local_idx, emb in enumerate(tqdm(embeddings[start_idx:end_idx], desc=f"Part {args.part_idx}")):
        global_idx = start_idx + local_idx
        if global_idx in stats:
            continue
        prompt_embeds = project_face_embs(pipeline, emb.unsqueeze(0).to(device=device, dtype=dtype))
        images, n_generated, n_no_face = sample_diverse_images(
            pipeline, face_analyzer, prompt_embeds, args.images_per_identity, args.batch_size,
            args.steps, args.guidance_scale, args.similarity_threshold, args.max_iters,
        )
        if len(images) < args.images_per_identity:
            tqdm.write(
                f"[WARN] identity {global_idx}: {len(images)}/{args.images_per_identity} diverse images "
                f"after {n_generated} generated ({n_no_face} without a face)"
            )
        for img_idx, img in enumerate(images):
            img.save(os.path.join(images_dir, f"{global_idx:06d}_{img_idx:02d}.png"))
        stats[global_idx] = {"n_accepted": len(images), "n_generated": n_generated, "n_no_face": n_no_face}
        write_part_metadata(meta_path, args, start_idx, end_idx, stats)

    print(f"Part {args.part_idx} done: {sum(v['n_accepted'] for v in stats.values())} images for {len(stats)} identities")


def cmd_merge(args):
    merged_dir = os.path.join(args.output_dir, "merged")
    merged_images_dir = os.path.join(merged_dir, "images")
    os.makedirs(merged_images_dir, exist_ok=True)

    all_stats, parts_found = {}, 0
    for part_idx in range(args.n_parts):
        part_dir = os.path.join(args.output_dir, f"part_{part_idx:04d}")
        meta_path = os.path.join(part_dir, "metadata.json")
        if not os.path.exists(meta_path):
            print(f"[WARN] part {part_idx}: {meta_path} not found, skipping")
            continue
        with open(meta_path) as f:
            part_stats = {int(k): v for k, v in json.load(f).get("stats", {}).items()}
        all_stats.update(part_stats)
        parts_found += 1

        images_dir = os.path.join(part_dir, "images")
        n_linked = 0
        for fname in os.listdir(images_dir) if os.path.isdir(images_dir) else []:
            src = os.path.abspath(os.path.join(images_dir, fname))
            dst = os.path.join(merged_images_dir, fname)
            if os.path.exists(dst):
                continue
            shutil.copy2(src, dst) if args.copy else os.symlink(src, dst)
            n_linked += 1
        print(f"Part {part_idx}: {len(part_stats)} identities, {n_linked} new files")

    meta = {
        "n_parts_merged": parts_found,
        "n_identities": len(all_stats),
        "n_images": sum(v["n_accepted"] for v in all_stats.values()),
        "stats": {str(k): v for k, v in all_stats.items()},
    }
    with open(os.path.join(merged_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Merged {meta['n_images']} images for {meta['n_identities']} identities -> {merged_images_dir}")


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    sp = sub.add_parser("sample", help="generate images for one partition of the identities")
    sp.add_argument("--embeddings_path", required=True, help="id_embeddings.pt, FloatTensor (N, 512)")
    sp.add_argument("--output_dir", required=True, help="parts are written to output_dir/part_XXXX/")
    sp.add_argument("--arc2face_models_dir", default="Arc2Face/models", help="contains encoder/ and arc2face/")
    sp.add_argument("--insightface_root", default="Arc2Face", help="contains models/antelopev2/")
    sp.add_argument("--part_idx", type=int, default=0)
    sp.add_argument("--n_parts", type=int, default=1)
    sp.add_argument("--images_per_identity", type=int, default=50)
    sp.add_argument("--batch_size", type=int, default=8, help="images per pipeline call")
    sp.add_argument("--steps", type=int, default=25, help="diffusion inference steps")
    sp.add_argument("--guidance_scale", type=float, default=3.0)
    sp.add_argument("--similarity_threshold", type=float, default=0.3,
                    help="max pairwise ArcFace cosine similarity between accepted images")
    sp.add_argument("--max_iters", type=int, default=50, help="max pipeline calls per identity")
    sp.add_argument("--device", default="cuda:0")

    mp = sub.add_parser("merge", help="merge part directories into output_dir/merged/")
    mp.add_argument("--output_dir", required=True)
    mp.add_argument("--n_parts", type=int, required=True)
    mp.add_argument("--copy", action="store_true", help="copy files instead of symlinking")
    return parser


def main():
    args = build_parser().parse_args()
    cmd_sample(args) if args.command == "sample" else cmd_merge(args)


if __name__ == "__main__":
    main()
