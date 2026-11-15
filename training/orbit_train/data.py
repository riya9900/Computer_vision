import random

import torch
from torch.utils.data import Dataset


def normalise(x: torch.Tensor) -> torch.Tensor:
    """Per-image, per-band standardisation (Sentinel-2 scales vary by scene)."""
    mean = x.mean(dim=(-2, -1), keepdim=True)
    std = x.std(dim=(-2, -1), keepdim=True) + 1e-6
    return (x - mean) / std


def split_pair(img: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """torchgeo OSCD gives (2, C, H, W) or (2C, H, W): before then after."""
    if img.ndim == 4:
        return img[0], img[1]
    before, after = img.chunk(2, dim=0)
    return before, after


class OSCDPatches(Dataset):
    """Random crops from OSCD pairs, with flip / rot90 / before-after-swap augmentation."""

    def __init__(self, root: str, split: str, patch: int = 96, n_per_epoch: int = 800,
                 augment: bool = True, download: bool = True):
        from torchgeo.datasets import OSCD

        self.ds = OSCD(root=root, split=split, bands=OSCD.all_bands, download=download)
        self.patch, self.augment, self.n = patch, augment, n_per_epoch

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        s = self.ds[random.randrange(len(self.ds))]
        before, after = split_pair(s["image"].float())
        mask = s["mask"].float()
        if mask.ndim == 2:  # torchgeo may return (H, W) or (1, H, W)
            mask = mask[None]
        _, h, w = before.shape
        y, x = random.randint(0, h - self.patch), random.randint(0, w - self.patch)
        sl = (slice(None), slice(y, y + self.patch), slice(x, x + self.patch))
        before, after, mask = before[sl], after[sl], mask[sl]
        if self.augment:
            if random.random() < 0.5:
                before, after, mask = (t.flip(-1) for t in (before, after, mask))
            k = random.randint(0, 3)
            before, after, mask = (torch.rot90(t, k, (-2, -1)) for t in (before, after, mask))
            if random.random() < 0.5:  # change is symmetric in time
                before, after = after, before
        return normalise(before), normalise(after), (mask > 0).float()
