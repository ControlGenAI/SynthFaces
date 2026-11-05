"""Config parsing, checkpoint loading and weight export for ID-Flow.

Two checkpoint kinds can be loaded by :func:`load_idflow`:

* an exported, self-describing weights file produced by ``scripts/export_weights.py``
  (``{"format": "idflow-v1", "hparams": ..., "state_dict": ...}``) - needs no config;
* a Lightning ``.ckpt`` written by ``scripts/train_idflow.py`` - hyper-parameters are stored
  inside; for checkpoints that lack them a training config JSON can be passed instead.
"""
import json

import torch

from .flow import DEFAULT_MODEL_PARAMS, IDFlow

EXPORT_FORMAT = "idflow-v1"


def load_config(path):
    with open(path, "r") as f:
        return json.load(f)


def hparams_from_config(cfg, num_train_steps=10_000):
    """Translate a training config dict into ``IDFlow`` keyword arguments."""
    mp = cfg["model_params"]
    return dict(
        model_params={k: mp[k] for k in DEFAULT_MODEL_PARAMS if k in mp},
        lr=mp.get("lr", 3e-4),
        loss_type=mp.get("loss_type", "l2"),
        guidance_scale=cfg.get("guidance_scale", 2.0),
        optimizer=cfg.get("optimizer", "adam"),
        lr_scheduler=cfg.get("lr_scheduler", "constant"),
        num_train_steps=num_train_steps,
    )


def _torch_load(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        # Lightning checkpoints may contain non-tensor python objects
        return torch.load(path, map_location="cpu", weights_only=False)


def _unet_state_dict(state_dict):
    return {k[len("model."):]: v for k, v in state_dict.items() if k.startswith("model.")}


def load_idflow(ckpt_path, config_path=None, device="cpu"):
    """Load a trained ID-Flow model (eval mode) from an exported file or a Lightning ckpt."""
    ckpt = _torch_load(ckpt_path)
    if ckpt.get("format") == EXPORT_FORMAT:
        hparams, unet_sd = ckpt["hparams"], ckpt["state_dict"]
    else:
        if "state_dict" not in ckpt:
            raise ValueError(f"{ckpt_path} is neither an exported ID-Flow file nor a Lightning checkpoint")
        if ckpt.get("hyper_parameters"):
            hparams = dict(ckpt["hyper_parameters"])
        elif config_path is not None:
            hparams = hparams_from_config(load_config(config_path))
        else:
            raise ValueError(
                "This checkpoint has no stored hyper-parameters; pass the training config (--model_config)."
            )
        unet_sd = _unet_state_dict(ckpt["state_dict"])
    model = IDFlow(**hparams)
    model.model.load_state_dict(unet_sd)
    return model.to(device).eval()


def export_weights(ckpt_path, out_path, config_path=None):
    """Write a weights-only, self-describing file that ``load_idflow`` (and therefore
    ``scripts/sample_identities.py``) can load without any config."""
    model = load_idflow(ckpt_path, config_path=config_path, device="cpu")
    payload = {
        "format": EXPORT_FORMAT,
        "hparams": dict(model.hparams),
        "state_dict": {k: v.detach().clone() for k, v in model.model.state_dict().items()},
    }
    torch.save(payload, out_path)
    return payload
