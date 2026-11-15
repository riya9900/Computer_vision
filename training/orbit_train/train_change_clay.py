"""Change head on cached frozen-Clay features, trained on OSCD.

    python -m orbit_train.train_change_clay --cache DIR --out DIR --head conv   --seed 0
    python -m orbit_train.train_change_clay --cache DIR --out DIR --head linear --seed 0   # linear probe baseline
"""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .clay import DIM, PATCH
from .metrics import change_metrics
from .train_change import dice_loss


class ConvChangeHead(nn.Module):
    """[f_before, f_after, |diff|] token maps -> conv -> pixel-shuffle x8 -> change logits."""

    def __init__(self, dim: int = DIM, hidden: int = 128, out_ch: int = 16, r: int = PATCH):
        super().__init__()
        self.drop = nn.Dropout2d(0.2)
        self.proj = nn.Sequential(nn.Conv2d(3 * dim, hidden, 1), nn.BatchNorm2d(hidden), nn.ReLU(True))
        self.conv = nn.Sequential(nn.Conv2d(hidden, hidden, 3, padding=1), nn.BatchNorm2d(hidden), nn.ReLU(True))
        self.ps_conv = nn.Conv2d(hidden, out_ch * r * r, 3, padding=1)
        self.ps = nn.PixelShuffle(r)
        self.out = nn.Sequential(nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.ReLU(True), nn.Conv2d(out_ch, 1, 1))

    def forward(self, fb, fa):
        x = self.drop(torch.cat([fb, fa, (fb - fa).abs()], 1))
        return self.out(self.ps(self.ps_conv(self.conv(self.proj(x)))))


class LinearProbe(nn.Module):
    """Baseline: one 1x1 conv on the same features, bilinear upsample."""

    def __init__(self, dim: int = DIM, r: int = PATCH):
        super().__init__()
        self.fc, self.r = nn.Conv2d(3 * dim, 1, 1), r

    def forward(self, fb, fa):
        x = self.fc(torch.cat([fb, fa, (fb - fa).abs()], 1))
        return F.interpolate(x, scale_factor=self.r, mode="bilinear", align_corners=False)


def load_scenes(path, device):
    scenes = torch.load(path)
    out = []
    for s in scenes:
        _, h, w = s["before"].shape
        mask = torch.zeros(h * PATCH, w * PATCH)
        mh, mw = s["mask"].shape
        mask[:mh, :mw] = s["mask"].float()
        out.append({"before": s["before"].to(device), "after": s["after"].to(device),
                    "mask": mask.to(device), "shape": (mh, mw)})
    return out


def sample_batch(scenes, bs, crop=12):
    fb, fa, mk = [], [], []
    for _ in range(bs):
        s = random.choice(scenes)
        _, h, w = s["before"].shape
        ty, tx = random.randint(0, h - crop), random.randint(0, w - crop)
        sl = (slice(None), slice(ty, ty + crop), slice(tx, tx + crop))
        b, a = s["before"][sl].float(), s["after"][sl].float()
        m = s["mask"][None, ty * PATCH:(ty + crop) * PATCH, tx * PATCH:(tx + crop) * PATCH]
        if random.random() < 0.5:
            b, a, m = b.flip(-1), a.flip(-1), m.flip(-1)
        if random.random() < 0.5:
            b, a = a, b  # change is symmetric in time
        fb.append(b); fa.append(a); mk.append(m)
    return torch.stack(fb), torch.stack(fa), torch.stack(mk)


@torch.no_grad()
def evaluate(head, scenes, thr=0.5):
    head.eval()
    preds, targets = [], []
    for s in scenes:
        mh, mw = s["shape"]
        logit = head(s["before"][None].float(), s["after"][None].float())[0, 0, :mh, :mw]
        preds.append((torch.sigmoid(logit) > thr).cpu().numpy().ravel())
        targets.append((s["mask"][:mh, :mw] > 0).cpu().numpy().ravel())
    return change_metrics(np.concatenate(preds), np.concatenate(targets))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--head", choices=["conv", "linear"], default="conv")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    train = load_scenes(Path(args.cache) / "oscd_train.pt", device)
    test = load_scenes(Path(args.cache) / "oscd_test.pt", device)

    head = (ConvChangeHead() if args.head == "conv" else LinearProbe()).to(device)
    n_params = sum(p.numel() for p in head.parameters())
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(8.0, device=device))

    history, t0 = [], time.time()
    steps = 50
    for ep in range(args.epochs):
        head.train()
        total = 0.0
        for _ in range(steps):
            fb, fa, m = sample_batch(train, 16)
            logits = head(fb, fa)
            loss = bce(logits, m) + dice_loss(logits, m)
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item()
        sched.step()
        history.append({"epoch": ep, "loss": total / steps})
        if ep % 5 == 0:
            print(history[-1], flush=True)
    train_seconds = time.time() - t0

    metrics = evaluate(head, test)
    name = f"clay_{args.head}"
    result = {"model": name, "seed": args.seed, "params": n_params,
              "frozen_clay_params": 311_000_000, "train_seconds": train_seconds,
              "device": device, "test": metrics, "history": history}
    torch.save(head.state_dict(), out / f"{name}_seed{args.seed}.pt")
    (out / f"{name}_seed{args.seed}.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "history"}, indent=2))


if __name__ == "__main__":
    main()
