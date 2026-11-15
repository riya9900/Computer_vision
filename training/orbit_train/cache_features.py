"""Run frozen Clay once and cache features, so heads train in minutes.

    python -m orbit_train.cache_features oscd    --encoder ENC.pt --data /content/oscd    --out DIR
    python -m orbit_train.cache_features eurosat --encoder ENC.pt --data /content/eurosat --out DIR
"""
import argparse
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .clay import band_indices, embed_scene, encode, load_encoder
from .data import split_pair


def cache_oscd(enc, root: str, out: Path, device: str):
    from torchgeo.datasets import OSCD

    idx = band_indices(OSCD.all_bands)
    for split in ("train", "test"):
        ds = OSCD(root=root, split=split, bands=OSCD.all_bands, download=True)
        scenes = []
        for i in range(len(ds)):
            s = ds[i]
            before, after = split_pair(s["image"].float())
            mask = s["mask"]
            mask = (mask[0] if mask.ndim == 3 else mask) > 0
            scenes.append({
                "before": embed_scene(enc, before[idx], device),
                "after": embed_scene(enc, after[idx], device),
                "mask": mask.to(torch.uint8),
            })
            print(split, i + 1, "/", len(ds), tuple(scenes[-1]["before"].shape), flush=True)
        torch.save(scenes, out / f"oscd_{split}.pt")


def cache_eurosat(enc, root: str, out: Path, device: str):
    from torchgeo.datasets import EuroSAT

    idx = band_indices(EuroSAT.all_band_names)
    for split in ("train", "val", "test"):
        ds = EuroSAT(root=root, split=split, bands=EuroSAT.all_band_names, download=True)
        loader = DataLoader(ds, batch_size=256, num_workers=2)
        cls, mean, labels = [], [], []
        for batch in loader:
            tok = encode(enc, batch["image"].float()[:, idx], device)
            cls.append(tok[:, 0].half().cpu())
            mean.append(tok[:, 1:].mean(1).half().cpu())
            labels.append(batch["label"])
        torch.save({"cls": torch.cat(cls), "mean": torch.cat(mean),
                    "labels": torch.cat(labels), "classes": list(ds.classes)},
                   out / f"eurosat_{split}.pt")
        print(split, len(ds), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=["oscd", "eurosat"])
    ap.add_argument("--encoder", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    enc = load_encoder(args.encoder, device)
    t0 = time.time()
    {"oscd": cache_oscd, "eurosat": cache_eurosat}[args.dataset](enc, args.data, out, device)
    print(f"cached {args.dataset} in {time.time() - t0:.0f}s on {device}")


if __name__ == "__main__":
    main()
