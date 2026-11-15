"""Classical baseline: change vector analysis on raw spectral bands (no learning).

Per scene: standardise each band per date, take the L2 norm of the band-difference
vector, threshold it. The threshold is picked on the train split only (max F1), then
applied unchanged to the test split.

    python -m orbit_train.baseline_spectral --root /content/oscd --out DIR
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .data import normalise, split_pair
from .metrics import change_metrics


def cva_magnitude(before: torch.Tensor, after: torch.Tensor) -> np.ndarray:
    diff = normalise(after) - normalise(before)
    return diff.pow(2).sum(0).sqrt().numpy()


def load(root: str, split: str):
    from torchgeo.datasets import OSCD

    ds = OSCD(root=root, split=split, bands=OSCD.all_bands, download=True)
    out = []
    for i in range(len(ds)):
        s = ds[i]
        before, after = split_pair(s["image"].float())
        mask = s["mask"]
        mask = (mask[0] if mask.ndim == 3 else mask) > 0
        out.append((cva_magnitude(before, after), mask.numpy()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    train, test = load(args.root, "train"), load(args.root, "test")
    mag_tr = np.concatenate([m.ravel() for m, _ in train])
    gt_tr = np.concatenate([g.ravel() for _, g in train])

    best_thr, best_f1 = 0.0, -1.0
    for thr in np.quantile(mag_tr, np.linspace(0.5, 0.99, 50)):
        f1 = change_metrics(mag_tr > thr, gt_tr)["f1"]
        if f1 > best_f1:
            best_thr, best_f1 = float(thr), f1

    metrics = change_metrics(np.concatenate([(m > best_thr).ravel() for m, _ in test]),
                             np.concatenate([g.ravel() for _, g in test]))
    result = {"model": "spectral_cva", "seed": 0, "params": 0, "train_seconds": 0.0,
              "threshold": best_thr, "train_f1": best_f1, "test": metrics}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "spectral_cva_seed0.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
