"""Export a trained ID-Flow checkpoint to a small, self-describing weights file.

    python scripts/export_weights.py --ckpt runs/idflow/checkpoints/last.ckpt --out idflow.pt

The exported file contains only the network weights and the model hyper-parameters (no
optimizer state), and can be passed directly to ``scripts/sample_identities.py --ckpt``
without any config.

For checkpoints that do not store hyper-parameters, pass the training config with --config.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import os

from idflow.io import export_weights


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True, help="Lightning checkpoint (.ckpt)")
    p.add_argument("--out", default="idflow.pt")
    p.add_argument("--config", default=None, help="training config JSON (only for checkpoints without hparams)")
    args = p.parse_args()

    payload = export_weights(args.ckpt, args.out, config_path=args.config)
    n_params = sum(v.numel() for v in payload["state_dict"].values())
    print(f"Wrote {args.out} ({os.path.getsize(args.out) / 1e6:.1f} MB, {n_params / 1e6:.2f}M parameters)")
    print("hparams:", payload["hparams"])


if __name__ == "__main__":
    main()
