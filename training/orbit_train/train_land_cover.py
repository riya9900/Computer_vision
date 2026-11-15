"""Land-cover head: 2-layer MLP on cached Clay features, trained on EuroSAT.

    python -m orbit_train.train_land_cover --cache DIR --out DIR --seed 0
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from torch import nn


def make_head(in_dim: int, n_classes: int) -> nn.Module:
    return nn.Sequential(nn.Linear(in_dim, 512), nn.ReLU(), nn.Dropout(0.25),
                         nn.Linear(512, n_classes))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--feature", choices=["cls", "mean"], default="cls")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cache, out = Path(args.cache), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tr, va, te = (torch.load(cache / f"eurosat_{s}.pt") for s in ("train", "val", "test"))
    classes = tr["classes"]

    def xy(d):
        return d[args.feature].float().to(device), d["labels"].long().to(device)

    (xtr, ytr), (xva, yva), (xte, yte) = xy(tr), xy(va), xy(te)
    mu, sd = xtr.mean(0, keepdim=True), xtr.std(0, keepdim=True) + 1e-6  # train-only stats
    xtr, xva, xte = ((x - mu) / sd for x in (xtr, xva, xte))

    head = make_head(xtr.shape[1], len(classes)).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    loss_fn = nn.CrossEntropyLoss()

    best, best_state, t0 = 0.0, None, time.time()
    for ep in range(args.epochs):
        head.train()
        perm = torch.randperm(len(xtr), device=device)
        for i in range(0, len(perm), 256):
            b = perm[i:i + 256]
            loss = loss_fn(head(xtr[b]), ytr[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
        head.eval()
        with torch.no_grad():
            val_acc = (head(xva).argmax(1) == yva).float().mean().item()
        if val_acc > best:  # model selection on validation only, never on test
            best, best_state = val_acc, {k: v.clone() for k, v in head.state_dict().items()}
        if ep % 10 == 0:
            print({"epoch": ep, "loss": round(loss.item(), 4), "val_acc": round(val_acc, 4)}, flush=True)
    train_seconds = time.time() - t0

    head.load_state_dict(best_state)
    head.eval()
    with torch.no_grad():
        pred = head(xte).argmax(1).cpu().numpy()
    true = yte.cpu().numpy()
    cm = confusion_matrix(true, pred, labels=list(range(len(classes))))
    result = {"model": "land_cover_mlp", "feature": args.feature, "seed": args.seed,
              "params": sum(p.numel() for p in head.parameters()),
              "train_seconds": train_seconds, "val_acc": best,
              "test_acc": float((pred == true).mean()), "classes": classes,
              "confusion_matrix": cm.tolist()}
    tag = f"land_cover_{args.feature}_seed{args.seed}"
    torch.save({"state": best_state, "mu": mu.cpu(), "sd": sd.cpu(), "classes": classes},
               out / f"{tag}.pt")
    (out / f"{tag}.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "confusion_matrix"}, indent=2))


if __name__ == "__main__":
    main()
