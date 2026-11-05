"""Conditional velocity network for ID-Flow.

An MLP U-Net with skip connections. Blocks are conditioned on time (sinusoidal embedding)
and on the labels (one embedding table per label, fed through AdaLN-style modulation).
Label embeddings contain an extra "null" class used for classifier-free guidance.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def get_activation_func(activation_type):
    if activation_type == "silu":
        return F.silu
    if activation_type == "leaky_relu":
        return F.leaky_relu
    if activation_type == "relu":
        return F.relu
    if activation_type == "sigmoid":
        return torch.sigmoid
    if activation_type == "gelu":
        return lambda x: F.gelu(x, approximate="tanh")
    if activation_type == "softsign":
        return F.softsign
    raise ValueError(f"Activation type {activation_type} not supported")


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.emb_dim = emb_dim

    def forward(self, t):
        half_dim = self.emb_dim // 2
        t = t.float().unsqueeze(-1)  # (B, 1)
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half_dim, device=t.device).float() / half_dim
        )
        sinusoid = t * freqs
        return torch.cat([torch.sin(sinusoid), torch.cos(sinusoid)], dim=-1)  # (B, emb_dim)


class LabelEmbedding(nn.Module):
    """Embedding table with one extra "null" class (index ``num_classes``) for CFG."""

    def __init__(self, num_classes, dim, dropout_prob=0.15):
        super().__init__()
        self.emb = nn.Embedding(num_classes + 1, dim)
        self.dropout_prob = dropout_prob
        self.num_classes = num_classes

    def forward(self, y, drop_labels=True):
        """y: (B,) class indices. If ``drop_labels``, each label is replaced by the null
        class with probability ``dropout_prob``."""
        if drop_labels and self.dropout_prob > 0:
            drop = torch.rand(y.shape[0], device=y.device) < self.dropout_prob
            y = torch.where(drop, self.num_classes, y)
        return self.emb(y)


def modulate(x, shift, scale):
    """Normalize x over the feature dim (B, D) and apply a conditional scale/shift."""
    x_mean, x_std = x.mean(dim=-1, keepdim=True), x.std(dim=-1, keepdim=True)
    return (x - x_mean) / (x_std + 1e-6) * scale + shift


class ModulatedResBlock(nn.Module):
    def __init__(self, input_dim, cond_dim, output_dim=None, dropout=0.1, activation_type="silu"):
        super().__init__()
        self.act_func = get_activation_func(activation_type)
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        self.adain_modulator = nn.Sequential(nn.SiLU(), nn.Linear(cond_dim, input_dim * 2))

        if output_dim is None:
            output_dim = input_dim
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.cond_dim = cond_dim

        self.fc1 = nn.Linear(input_dim, input_dim)
        self.fc2 = nn.Linear(input_dim, output_dim)
        self.time_proj = nn.Linear(512, input_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, c, t_emb):
        assert x.dim() == 2, f"x should have shape (B, D), but got {tuple(x.shape)}"
        h = self.norm1(x)
        h = self.act_func(h)
        h = self.fc1(h)
        h = h + self.time_proj(t_emb)
        h = self.dropout(h)
        h = x + h
        h = self.norm2(h)

        shift, scale = self.adain_modulator(c).chunk(2, dim=-1)
        h = modulate(h, shift, scale)

        h = self.act_func(h)
        h = self.fc2(h)
        h = self.dropout(h)
        return h


class ConditionalLatent1DUNet(nn.Module):
    """Label- and time-conditioned MLP U-Net operating on 512-d identity embeddings."""

    def __init__(
        self,
        dim=512,
        depth=2,
        hidden_dims=None,
        class_nums=(9, 7, 2),
        c_embed_dim=512,
        activation_type="gelu",
        scale_t=True,
    ):
        super().__init__()
        self.time_embed = SinusoidalTimeEmbedding(512)
        self.scale_t = scale_t
        self.depth = depth
        self.class_nums = list(class_nums)
        self.modality_cnt = len(self.class_nums)
        self.dim = dim
        self.c_embed_dim = c_embed_dim
        self.activation_type = activation_type

        if self.modality_cnt > 1:
            self.label_embedding = nn.ModuleList(
                [LabelEmbedding(n, c_embed_dim) for n in self.class_nums]
            )
        else:
            self.label_embedding = LabelEmbedding(self.class_nums[0], c_embed_dim)

        if hidden_dims is None:
            hidden_dims = [dim] * (2 * depth)
        assert len(hidden_dims) == 2 * depth, "hidden_dims must have length 2*depth"
        cond_dim = self.modality_cnt * c_embed_dim

        self.encoder = nn.ModuleList(
            [
                ModulatedResBlock(
                    hidden_dims[i], cond_dim, output_dim=hidden_dims[i + 1],
                    activation_type=activation_type,
                )
                for i in range(depth)
            ]
        )
        self.middle = nn.ModuleList(
            [
                ModulatedResBlock(
                    hidden_dims[depth], cond_dim, output_dim=hidden_dims[depth],
                    activation_type=activation_type,
                )
            ]
        )
        self.decoder = nn.ModuleList(
            [
                ModulatedResBlock(
                    hidden_dims[depth + i], cond_dim,
                    output_dim=hidden_dims[depth + 1 + i if i < depth - 1 else 0],
                    activation_type=activation_type,
                )
                for i in range(depth)
            ]
        )

    def embed_labels(self, y, drop_labels=True):
        if self.modality_cnt > 1:
            ys = torch.chunk(y, self.modality_cnt, dim=-1)
            c = [emb(yi.squeeze(-1), drop_labels) for emb, yi in zip(self.label_embedding, ys)]
            return torch.cat(c, dim=-1)
        return self.label_embedding(y[:, 0], drop_labels)

    def forward(self, x, y, t, drop_labels=True):
        """
        x: (B, 512) points on the sphere, y: (B, n_labels) long, t: (B,) in [0, 1].

        ``drop_labels=True`` (default, used for training) randomly replaces each label by
        the null class with probability 0.15 so that the model also learns the
        unconditional velocity needed for classifier-free guidance. Set it to ``False``
        for sampling.
        """
        if self.scale_t:
            t = t * 1000.0
        t_emb = self.time_embed(t)
        c = self.embed_labels(y, drop_labels)

        h = x
        hs = []
        for layer in self.encoder:
            h = layer(h, c, t_emb)
            hs.append(h)
        for layer in self.middle:
            h = layer(h, c, t_emb)
        for layer in self.decoder:
            h = layer(h, c, t_emb)
            h = h + hs.pop()
        assert h.shape == x.shape
        return h
