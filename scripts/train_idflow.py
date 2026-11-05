"""Train ID-Flow on a set of labelled ArcFace identity embeddings.

    python scripts/train_idflow.py --config configs/idflow_best.json --data_path data/embeddings
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import lightning as L
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

from idflow.data import IdentityDataModule
from idflow.flow import IDFlow
from idflow.io import hparams_from_config, load_config


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/idflow_best.json", help="training config (JSON)")
    p.add_argument("--data_path", required=True, help="dir with embeddings.pt and labels.pt")
    p.add_argument("--val_data_path", default=None,
                   help="optional dir with a fixed validation set (embeddings.pt, labels.pt); "
                        "if omitted, val_fraction of --data_path is held out")
    p.add_argument("--output_dir", default="runs/idflow", help="checkpoints and logs are written here")
    p.add_argument("--device_id", type=int, default=0, help="GPU index (ignored without CUDA)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=None, help="override config epochs")
    p.add_argument("--resume_from_ckpt", default=None, help="Lightning checkpoint to resume from")
    p.add_argument("--wandb_project", default=None, help="log to Weights & Biases (needs `wandb login`)")
    p.add_argument("--run_name", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    L.seed_everything(args.seed)

    epochs = args.epochs or cfg["epochs"]
    data = IdentityDataModule(args.data_path, val_data_path=args.val_data_path, **cfg["data_params"])
    data.setup()
    num_train_steps = epochs * len(data.train_data) // cfg["data_params"]["batch_size"]
    model = IDFlow(**hparams_from_config(cfg, num_train_steps=num_train_steps))

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    if args.wandb_project:
        from lightning.pytorch.loggers import WandbLogger

        logger = WandbLogger(project=args.wandb_project, name=args.run_name, save_dir=args.output_dir)
        logger.experiment.config.update(cfg)
    else:
        logger = CSVLogger(args.output_dir, name="logs")

    ckpt_cb = ModelCheckpoint(
        dirpath=os.path.join(args.output_dir, "checkpoints"),
        filename="epoch={epoch}-step={step}",
        auto_insert_metric_name=False,
        every_n_epochs=cfg.get("check_every_n_epochs", 100),
        save_top_k=-1,
        save_last=True,
    )
    use_gpu = torch.cuda.is_available()
    trainer = L.Trainer(
        max_epochs=epochs,
        accelerator="gpu" if use_gpu else "cpu",
        devices=[args.device_id] if use_gpu else 1,
        log_every_n_steps=50,
        num_sanity_val_steps=0,
        gradient_clip_val=1.0,
        callbacks=[ckpt_cb],
        logger=logger,
        default_root_dir=args.output_dir,
    )
    trainer.fit(model, datamodule=data, ckpt_path=args.resume_from_ckpt)


if __name__ == "__main__":
    main()
