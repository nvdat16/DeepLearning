"""Ghép ảnh kết quả phát hiện của các model thành hình định tính cho report.

    python make_qualitative.py

Lưu Report/latex/figures/qualitative.jpg (lưới: mỗi hàng một ảnh test, mỗi cột một model).
Nếu thư mục ảnh của một model chưa có, cột đó dùng ảnh thay thế trong MOCKUP
và được đóng dấu "MOCKUP" màu đỏ. Đặt ảnh thật vào thư mục là tự thay.
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

ROOT = os.path.dirname(os.path.abspath(__file__))
# (tên hiển thị, thư mục ảnh dự đoán) theo thứ tự cột
MODELS = [
    ("Faster R-CNN", "FasterR-CNN"),
    ("RT-DETR", "rtdetr"),
    ("YOLOv12m", "yolo12m"),
    ("YOLO26m", "yolo26m"),
]
IMAGES = ["0000272_03500_d_0000008.jpg", "9999938_00000_d_0000174.jpg"]
# ảnh thay thế tạm khi thiếu thư mục: {model: [thư mục cho hàng 1, hàng 2, ...]}
MOCKUP = {"RT-DETR": ["yolo12m", "yolo26m"]}
OUT = os.path.join(ROOT, "Report", "latex", "figures", "qualitative.jpg")
# True: đóng dấu "MOCKUP" lên ô dùng ảnh thay thế; False: không đóng dấu (chỉ in cảnh báo)
STAMP_MOCKUP = False


def load(model, folder, row):
    path = os.path.join(ROOT, folder, IMAGES[row])
    if os.path.exists(path):
        return mpimg.imread(path), False
    stand_in = os.path.join(ROOT, MOCKUP[model][row], IMAGES[row])
    return mpimg.imread(stand_in), True


plt.rcParams.update({"font.family": "serif", "font.size": 11})
fig, axes = plt.subplots(len(IMAGES), len(MODELS), figsize=(4 * len(MODELS), 2.35 * len(IMAGES)),
                         gridspec_kw=dict(wspace=0.02, hspace=0.02))
for c, (name, folder) in enumerate(MODELS):
    for r in range(len(IMAGES)):
        ax = axes[r][c]
        img, is_mockup = load(name, folder, r)
        ax.imshow(img)
        ax.set_axis_off()
        if is_mockup:
            print(f"WARNING: {name} / {IMAGES[r]} dùng ảnh thay thế, không phải kết quả thật của {name}")
        if is_mockup and STAMP_MOCKUP:
            ax.text(0.5, 0.5, "MOCKUP", transform=ax.transAxes, ha="center", va="center",
                    fontsize=30, color="red", alpha=0.75, fontweight="bold")
        if r == 0:
            ax.set_title(name, pad=4)

fig.savefig(OUT, dpi=200, bbox_inches="tight", pad_inches=0.02,
            pil_kwargs={"quality": 88})
print(f"Saved {OUT}")
