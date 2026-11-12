"""Visualize phân bố kích thước bbox của VisDrone2019-DET (train/val/test-dev).

    python bbox_stats.py            # lưu bbox_stats_<split>.png và bbox_stats_all.png
"""
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
SPLITS = ["train", "val", "test-dev"]
COCO_BINS = [0, 32 ** 2, 96 ** 2, np.inf]
COCO_LABELS = ["Small (<32²)", "Medium (32²–96²)", "Large (≥96²)"]
COLORS = ["#d95f02", "#1b9e77", "#7570b3"]


def load_wh(split):
    """[w, h] của 10 lớp mục tiêu có score>0; bỏ category 0/11 và bbox rỗng."""
    rows = []
    for f in glob.glob(os.path.join(ROOT, f"VisDrone2019-DET-{split}", "annotations", "*.txt")):
        a = np.loadtxt(f, delimiter=",", ndmin=2, usecols=range(8))
        if a.size:
            target = (a[:, 5] >= 1) & (a[:, 5] <= 10) & (a[:, 4] > 0)
            rows.append(a[target][:, 2:4])
    wh = np.concatenate(rows)
    return wh[(wh > 0).all(axis=1)]


def plot(wh, title, path):
    w, h = wh[:, 0], wh[:, 1]
    area = w * h
    counts = np.histogram(area, COCO_BINS)[0]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    fig.suptitle(f"{title} — {len(wh):,} bbox", fontsize=14)

    ax = axes[0]
    idx = np.random.default_rng(0).choice(len(wh), min(len(wh), 30000), replace=False)
    ax.scatter(w[idx], h[idx], s=3, alpha=0.2, color=COLORS[0], rasterized=True)
    lim = max(np.percentile(w, 99.9), np.percentile(h, 99.9))
    ax.plot([0, lim], [0, lim], "k--", lw=0.8, label="w = h")
    ax.set(xlim=(0, lim), ylim=(0, lim), xlabel="Width (px)", ylabel="Height (px)",
           title="BBox Width–Height Distribution")
    ax.legend()

    ax = axes[1]
    bins = np.logspace(np.log10(area.min()), np.log10(area.max()), 60)
    ax.hist(area, bins=bins, color=COLORS[1], edgecolor="white", lw=0.3)
    ax.set_xscale("log")
    for t in COCO_BINS[1:3]:
        ax.axvline(t, color="k", ls="--", lw=0.8)
    ax.axvline(np.median(area), color="red", lw=1, label=f"median = {np.median(area):.0f} px²")
    ax.set(xlabel="Area (px², log scale)", ylabel="Số bbox", title="BBox Area Distribution")
    ax.legend()

    ax = axes[2]
    bars = ax.bar(COCO_LABELS, counts / counts.sum() * 100, color=COLORS)
    for b, c in zip(bars, counts):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.5,
                f"{c / counts.sum() * 100:.1f}%\n({c:,})", ha="center")
    ax.set(ylabel="Tỉ lệ (%)", ylim=(0, max(counts / counts.sum() * 100) * 1.2),
           title="Object Size Distribution (COCO)")

    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"{path}: small/medium/large = {counts.tolist()}")


if __name__ == "__main__":
    data = {s: load_wh(s) for s in SPLITS}
    for s, wh in data.items():
        plot(wh, f"VisDrone2019-DET {s}", os.path.join(ROOT, f"bbox_stats_{s}.png"))
    plot(np.concatenate(list(data.values())), "VisDrone2019-DET (all splits)",
         os.path.join(ROOT, "bbox_stats_all.png"))
