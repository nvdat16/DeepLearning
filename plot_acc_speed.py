"""Vẽ biểu đồ mAP50:95 theo FPS (PyTorch và TensorRT) cho report.

    python plot_acc_speed.py

Lưu Report/latex/figures/acc_speed.png. Sửa RESULTS nếu có số liệu mới.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker

# model: (mAP50:95 %, FPS PyTorch, FPS TensorRT, màu)
RESULTS = {
    "Faster R-CNN": (19.4, 20, 38, "#8a8986"),   # baseline cổ điển: xám trung tính
    "RT-DETR":      (21.3, 51, 198, "#2a78d6"),
    "YOLOv12m":     (21.1, 48, 143, "#eb6834"),
    "YOLO26m":      (21.7, 53, 208, "#1baf7a"),
}
# vị trí nhãn tên (dx, dy theo điểm ảnh) cạnh điểm TensorRT, tránh chồng nhau
LABEL_OFFSET = {name: (7, 0) for name in RESULTS}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "Report", "latex", "figures", "acc_speed.png")

plt.rcParams.update({"font.family": "serif", "font.size": 8})
fig, ax = plt.subplots(figsize=(3.3, 2.4))

for name, (ap, fps_pt, fps_trt, color) in RESULTS.items():
    ax.annotate("", xy=(fps_trt, ap), xytext=(fps_pt, ap),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=1.2,
                                shrinkA=5, shrinkB=5, mutation_scale=7))
    ax.scatter(fps_pt, ap, s=36, facecolor="white", edgecolor=color, linewidth=1.6, zorder=3)
    ax.scatter(fps_trt, ap, s=36, color=color, edgecolor="white", linewidth=1.2, zorder=3)
    dx, dy = LABEL_OFFSET[name]
    ax.annotate(name, (fps_trt, ap), xytext=(dx, dy), textcoords="offset points",
                ha="left" if dx > 0 else "right", va="center", color=INK, fontsize=7.5)

ax.set_xscale("log")
ax.set_xticks([20, 50, 100, 200])
ax.set_xticklabels(["20", "50", "100", "200"])
ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
ax.set_xlim(15, 600)
ax.set_ylim(19.0, 22.0)
ax.set_xlabel("Speed (FPS, log scale)", color=MUTED)
ax.set_ylabel("mAP$_{50:95}$ (%)", color=MUTED)
ax.tick_params(colors=MUTED, length=0)
ax.grid(color=GRID, linewidth=0.6)
ax.set_axisbelow(True)
for s in ax.spines.values():
    s.set_visible(False)

# chú thích kiểu điểm: rỗng = PyTorch, đặc = TensorRT
h_pt = ax.scatter([], [], s=30, facecolor="white", edgecolor=MUTED, linewidth=1.4)
h_trt = ax.scatter([], [], s=30, color=MUTED)
ax.legend([h_pt, h_trt], ["PyTorch", "TensorRT"], loc="lower right", frameon=False,
          handletextpad=0.2, labelcolor=MUTED)

fig.tight_layout(pad=0.3)
os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, dpi=300)
print(f"Saved {OUT}")
