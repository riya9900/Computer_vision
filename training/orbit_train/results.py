"""Merge every result JSON in a folder into the final tables (mean +/- std over seeds).

    python -m orbit_train.results --dir /content/drive/MyDrive/orbitwatch

Writes results.md (change-detection table + land-cover table) and a confusion-matrix PNG.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

CHANGE_ORDER = [("spectral_cva", "Spectral differencing (CVA)"),
                ("clay_linear", "Frozen Clay + linear probe"),
                ("clay_conv", "Frozen Clay + conv head (ours)"),
                ("siamese_unet", "Siamese U-Net from scratch (ours)")]


def ms(vals):
    return f"{np.mean(vals):.3f} ± {np.std(vals):.3f}" if len(vals) > 1 else f"{vals[0]:.3f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    args = ap.parse_args()
    d = Path(args.dir)
    runs = [json.loads(p.read_text()) for p in sorted(d.glob("*.json"))]

    groups = defaultdict(list)
    for r in runs:
        groups[r["model"]].append(r)

    lines = ["## Change detection (OSCD test, 10 pairs)", "",
             "| Model | Seeds | F1 | IoU | Precision | Recall | Trainable params | Train time (s) |",
             "|---|---|---|---|---|---|---|---|"]
    for key, label in CHANGE_ORDER:
        rs = groups.get(key)
        if not rs:
            continue
        col = lambda m: ms([r["test"][m] for r in rs])  # noqa: E731
        lines.append(f"| {label} | {len(rs)} | {col('f1')} | {col('iou')} | {col('precision')} | "
                     f"{col('recall')} | {rs[0]['params']:,} | {np.mean([r['train_seconds'] for r in rs]):.0f} |")

    lc = [r for r in runs if r["model"] == "land_cover_mlp"]
    if lc:
        lines += ["", "## Land cover (EuroSAT held-out test split)", "",
                  "| Features | Seeds | Test accuracy | Trainable params |", "|---|---|---|---|"]
        for feat in sorted({r["feature"] for r in lc}):
            rs = [r for r in lc if r["feature"] == feat]
            lines.append(f"| Clay {feat} token + MLP | {len(rs)} | {ms([r['test_acc'] for r in rs])} | {rs[0]['params']:,} |")

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        r0 = sorted(lc, key=lambda r: (r["feature"] != "cls", r["seed"]))[0]
        cm = np.array(r0["confusion_matrix"], dtype=float)
        cm = cm / cm.sum(1, keepdims=True)
        fig, ax = plt.subplots(figsize=(8, 7))
        ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(len(r0["classes"])), r0["classes"], rotation=60, ha="right")
        ax.set_yticks(range(len(r0["classes"])), r0["classes"])
        for i in range(len(cm)):
            for j in range(len(cm)):
                ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center",
                        color="white" if cm[i, j] > 0.5 else "black", fontsize=7)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        ax.set_title(f"EuroSAT confusion matrix (Clay {r0['feature']}, seed {r0['seed']})")
        fig.tight_layout()
        fig.savefig(d / "eurosat_confusion.png", dpi=150)

    text = "\n".join(lines) + "\n"
    (d / "results.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
