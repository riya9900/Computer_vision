"""Train the Siamese U-Net on OSCD and evaluate on the 10 test pairs.

Colab:  python -m orbit_train.train_change --root /content/oscd --out /content/drive/MyDrive/orbitwatch
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .data import OSCDPatches, normalise, split_pair
from .metrics import change_metrics
from .models import SiameseUNet


def dice_loss(logits, target):
    p = torch.sigmoid(logits)
    inter = (p * target).sum()
    return 1 - (2 * inter + 1) / (p.sum() + target.sum() + 1)


@torch.no_grad()
def evaluate(model, ds, device, thr=0.5, tile=256):
    """Full-image evaluation, tiled so large scenes fit in memory."""
    model.eval()
    preds, targets = [], []
    for s in ds.samples:
        before, after = split_pair(s["image"].float())
        mask = s["mask"]
        _, h, w = before.shape
        out = np.zeros((h, w), dtype=np.uint8)
        for y in range(0, h, tile):
            for x in range(0, w, tile):
                b = normalise(before[:, y:y + tile, x:x + tile])[None].to(device)
                a = normalise(after[:, y:y + tile, x:x + tile])[None].to(device)
                th, tw = b.shape[-2:]
                ph, pw = (-th) % 8, (-tw) % 8
                b, a = (nn.functional.pad(t, (0, pw, 0, ph), mode="reflect") for t in (b, a))
                prob = torch.sigmoid(model(b, a))[0, 0, :th, :tw].cpu().numpy()
                out[y:y + th, x:x + tw] = prob > thr
        preds.append(out.ravel())
        targets.append((mask.numpy() > 0).ravel())
    return change_metrics(np.concatenate(preds), np.concatenate(targets))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    train = OSCDPatches(args.root, "train")
    test = OSCDPatches(args.root, "test", augment=False)
    loader = DataLoader(train, batch_size=16, num_workers=2)

    model = SiameseUNet().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(8.0, device=device))  # changes are rare

    history, t0 = [], time.time()
    for ep in range(args.epochs):
        model.train()
        total = 0.0
        for before, after, mask in loader:
            before, after, mask = before.to(device), after.to(device), mask.to(device)
            logits = model(before, after)
            loss = bce(logits, mask) + dice_loss(logits, mask)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
        sched.step()
        history.append({"epoch": ep, "loss": total / len(loader)})
        print(history[-1], flush=True)

    train_seconds = time.time() - t0
    metrics = evaluate(model, test, device)
    result = {"model": "siamese_unet", "seed": args.seed, "params": n_params,
              "train_seconds": train_seconds, "device": device, "test": metrics, "history": history}
    torch.save(model.state_dict(), out / f"siamese_unet_seed{args.seed}.pt")
    (out / f"siamese_unet_seed{args.seed}.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "history"}, indent=2))


if __name__ == "__main__":
    main()
