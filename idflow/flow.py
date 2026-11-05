"""ID-Flow: label-conditional Riemannian flow matching on the unit sphere of ArcFace embeddings.

Training follows geodesic conditional flow matching (https://arxiv.org/abs/2412.06264):
x_0 is uniform on the sphere, x_1 is a data embedding, x_t lies on the geodesic between
them and the network regresses the (tangent) velocity. Sampling integrates the learned
ODE with Euler steps and classifier-free guidance, projecting back to the sphere each step.
"""
import math

import torch
import torch.nn.functional as F
import lightning as L
from flow_matching.path import GeodesicProbPath
from flow_matching.path.scheduler import CondOTScheduler
from flow_matching.utils.manifolds import Sphere

from .labels import CLASS_NUMS
from .model import ConditionalLatent1DUNet

DEFAULT_MODEL_PARAMS = dict(
    depth=2,
    hidden_dims=None,
    c_embed_dim=512,
    activation_type="gelu",
    class_nums=list(CLASS_NUMS),
)


def get_loss(loss_type):
    if loss_type == "l2":
        return F.mse_loss
    if loss_type == "cosine":
        return lambda pred, targ: 1 - F.cosine_similarity(pred, targ, dim=-1).mean()
    raise ValueError(f"Unsupported loss type: {loss_type}")


def random_sphere_points(shape, device=None, generator=None):
    x = torch.randn(shape, device=device, generator=generator)
    return x / torch.norm(x, dim=-1, keepdim=True)


class IDFlow(L.LightningModule):
    def __init__(
        self,
        model_params=None,
        lr=3e-4,
        loss_type="l2",
        guidance_scale=2.0,
        optimizer="adam",
        lr_scheduler="constant",
        num_train_steps=10_000,
    ):
        super().__init__()
        model_params = {**DEFAULT_MODEL_PARAMS, **(model_params or {})}
        self.save_hyperparameters(
            dict(
                model_params=model_params, lr=lr, loss_type=loss_type,
                guidance_scale=guidance_scale, optimizer=optimizer,
                lr_scheduler=lr_scheduler, num_train_steps=num_train_steps,
            )
        )
        self.model = ConditionalLatent1DUNet(scale_t=True, **model_params)
        self.class_nums = list(model_params["class_nums"])
        self.lr = lr
        self.loss_type = loss_type
        self.guidance_scale = guidance_scale
        self.optimizer_name = optimizer
        self.lr_scheduler_name = lr_scheduler
        self.num_train_steps = num_train_steps

        self.manifold = Sphere()
        self.path = GeodesicProbPath(manifold=self.manifold, scheduler=CondOTScheduler())

    # ------------------------------------------------------------------ training
    def forward(self, x, y, t):
        return self.model(x=x, y=y, t=t)

    def _flow_loss(self, batch):
        x_1, y = batch
        t = torch.rand((x_1.shape[0],), device=self.device)
        x_0 = random_sphere_points(x_1.shape, device=x_1.device).to(x_1.dtype)
        sample = self.path.sample(t=t, x_0=x_0, x_1=x_1)
        pred = self.model(x=sample.x_t, y=y, t=t)
        pred = self.manifold.proju(sample.x_t, pred)  # project onto the tangent space
        return get_loss(self.loss_type)(pred, sample.dx_t)

    def training_step(self, batch, batch_idx):
        loss = self._flow_loss(batch)
        self.log("train/loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._flow_loss(batch)
        self.log("val/loss", loss)
        return loss

    def configure_optimizers(self):
        if self.optimizer_name == "adam":
            optim = torch.optim.Adam(self.parameters(), lr=self.lr)
        elif self.optimizer_name == "adam-w":
            optim = torch.optim.AdamW(self.parameters(), lr=self.lr)
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer_name}")
        if self.lr_scheduler_name == "constant":
            return optim
        if self.lr_scheduler_name == "cosine":
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=self.num_train_steps)
            return {"optimizer": optim, "lr_scheduler": {"scheduler": sched, "interval": "step"}}
        raise ValueError(f"Unknown lr scheduler: {self.lr_scheduler_name}")

    # ------------------------------------------------------------------ sampling
    @torch.no_grad()
    def sample(self, bs, y, guidance_scale=None, step_size=1 / 100, x_init=None, device=None):
        """Sample ``bs`` identity embeddings (unit-norm, shape (bs, 512)).

        y: (bs, 3) long tensor of [age, race, gender] class indices.
        guidance_scale: classifier-free guidance weight (default: ``self.guidance_scale``).
        """
        if x_init is None:
            device = device if device is not None else self.device
            x_init = random_sphere_points((bs, 512), device=device)
        else:
            assert x_init.shape[0] == bs, f"x_init should have batch size {bs}"
            if device is not None:
                x_init = x_init.to(device)
        if guidance_scale is None:
            guidance_scale = self.guidance_scale

        # the null label for every attribute is its class count
        uncond_y = torch.tensor(self.class_nums, device=x_init.device, dtype=torch.long)
        uncond_y = uncond_y.unsqueeze(0).repeat(bs, 1)
        y = y.to(x_init.device)

        n_steps = math.ceil(1.0 / step_size)
        grid = torch.tensor([step_size * i for i in range(n_steps)] + [1.0], device=x_init.device)

        def velocity(x, labels, t):
            # no random label dropout at inference; the null label is passed explicitly for CFG
            return self.manifold.proju(x, self.model(x=x, y=labels, t=t, drop_labels=False))

        xt = x_init
        for t0, t1 in zip(grid[:-1], grid[1:]):
            dt = t1 - t0
            t = t0.repeat(bs)
            v_cond = velocity(xt, y, t)
            v_uncond = velocity(xt, uncond_y, t)
            v = v_uncond + guidance_scale * (v_cond - v_uncond)
            xt = self.manifold.projx(xt + dt * v)
        return xt
