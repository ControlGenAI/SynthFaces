"""Data loading for ID-Flow training.

Expected data layout (a directory):

    embeddings.pt   FloatTensor (N, 512)  L2-normalised ArcFace identity embeddings
    labels.pt       LongTensor  (N, 3)    class indices, columns = [age, race, gender]
                                          (see ``idflow.labels`` for the vocabularies)
"""
import os

import torch
import lightning as L
from torch.utils.data import DataLoader, TensorDataset

from .labels import CLASS_NUMS


@torch.no_grad()
def slerp_interpolation(e1, e2, t):
    """Spherical linear interpolation between rows of e1 and e2. t: (B,)"""
    e1 = e1 / e1.norm(dim=-1, keepdim=True)
    e2 = e2 / e2.norm(dim=-1, keepdim=True)
    dot = torch.clamp(torch.einsum("nd,nd->n", e1, e2), -1.0, 1.0)
    omega = torch.acos(dot)
    sin_omega = torch.sin(omega)
    out = torch.empty_like(e1)
    same = sin_omega.abs() < 5e-3  # (almost) identical points: nothing to interpolate
    out[same] = e1[same]
    if (~same).any():
        a = torch.sin((1 - t[~same]) * omega[~same]) / sin_omega[~same]
        b = torch.sin(t[~same] * omega[~same]) / sin_omega[~same]
        out[~same] = a[:, None] * e1[~same] + b[:, None] * e2[~same]
    return out


class RandomInterpolationTransform:
    """Slerp each embedding of a batch towards a random other embedding of the batch by a
    random fraction in [0, max_t). Labels are kept from the original embedding."""

    def __init__(self, max_t=0.3):
        self.max_t = max_t

    def __call__(self, x):
        if x.shape[0] < 2:
            return x
        t = torch.rand((x.shape[0],), device=x.device) * self.max_t
        return slerp_interpolation(x, x[torch.randperm(x.shape[0])], t)


def load_embeddings_and_labels(data_path):
    emb = torch.load(os.path.join(data_path, "embeddings.pt"), map_location="cpu").float()
    labels = torch.load(os.path.join(data_path, "labels.pt"), map_location="cpu").long()
    assert emb.dim() == 2 and emb.shape[1] == 512, f"embeddings must be (N, 512), got {tuple(emb.shape)}"
    assert labels.shape == (emb.shape[0], len(CLASS_NUMS)), (
        f"labels must be (N, {len(CLASS_NUMS)}), got {tuple(labels.shape)}"
    )
    for i, n in enumerate(CLASS_NUMS):
        assert 0 <= labels[:, i].min() and labels[:, i].max() < n, f"label column {i} out of range"
    return emb, labels


class IdentityDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_path,
        batch_size=128,
        interpolation_aug=True,
        val_fraction=0.15,
        seed=42,
        num_workers=0,
        val_data_path=None,
    ):
        super().__init__()
        self.data_path = data_path
        self.val_data_path = val_data_path
        self.batch_size = batch_size
        self.val_fraction = val_fraction
        self.seed = seed
        self.num_workers = num_workers
        self.train_transform = RandomInterpolationTransform(max_t=0.3) if interpolation_aug else None

    def setup(self, stage=None):
        emb, labels = load_embeddings_and_labels(self.data_path)
        if self.val_data_path is not None:
            # explicit validation set: all of data_path is used for training
            val_emb, val_labels = load_embeddings_and_labels(self.val_data_path)
            self.train_data = TensorDataset(emb, labels)
            self.val_data = TensorDataset(val_emb, val_labels)
        else:
            g = torch.Generator().manual_seed(self.seed)
            perm = torch.randperm(emb.shape[0], generator=g)
            n_val = int(round(emb.shape[0] * self.val_fraction))
            val_idx, train_idx = perm[:n_val], perm[n_val:]
            self.train_data = TensorDataset(emb[train_idx], labels[train_idx])
            self.val_data = TensorDataset(emb[val_idx], labels[val_idx])
        print(f"Train identities: {len(self.train_data)} | validation identities: {len(self.val_data)}")

    def _collate_train(self, batch):
        x = torch.stack([b[0] for b in batch])
        y = torch.stack([b[1] for b in batch])
        if self.train_transform is not None:
            x = self.train_transform(x)
        return x, y

    def train_dataloader(self):
        return DataLoader(
            self.train_data, batch_size=self.batch_size, shuffle=True,
            num_workers=self.num_workers, collate_fn=self._collate_train,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_data, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers
        )
