"""Chuyển VisDrone2019-DET sang định dạng YOLO và tạo file cấu hình dataset.

    python src/prepare_data.py --root . --out datasets/visdrone
"""
import argparse
from pathlib import Path

from PIL import Image

CLASSES = ["pedestrian", "people", "bicycle", "car", "van", "truck",
           "tricycle", "awning-tricycle", "bus", "motor"]
# thư mục gốc -> tên tập trong dataset YOLO
SPLITS = {"train": "train", "val": "val", "test-dev": "test"}


def convert_split(src, dst):
    (dst / "images").mkdir(parents=True, exist_ok=True)
    (dst / "labels").mkdir(parents=True, exist_ok=True)
    n_box = 0
    for img in sorted((src / "images").glob("*.jpg")):
        link = dst / "images" / img.name
        if not link.exists():
            link.symlink_to(img.resolve())
        w, h = Image.open(img).size
        lines = []
        ann = src / "annotations" / f"{img.stem}.txt"
        for row in ann.read_text().splitlines() if ann.exists() else []:
            p = row.strip().split(",")
            x, y, bw, bh, score, cat = (int(v) for v in p[:6])
            # bỏ ignored (0), others (11), hộp lỗi / score = 0
            if cat < 1 or cat > 10 or score == 0 or bw <= 0 or bh <= 0:
                continue
            x1, y1 = max(x, 0), max(y, 0)
            x2, y2 = min(x + bw, w), min(y + bh, h)
            if x2 <= x1 or y2 <= y1:
                continue
            lines.append(f"{cat - 1} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} "
                         f"{(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}")
        (dst / "labels" / f"{img.stem}.txt").write_text("\n".join(lines))
        n_box += len(lines)
    return n_box


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".", help="thư mục chứa VisDrone2019-DET-*")
    ap.add_argument("--out", default="datasets/visdrone")
    args = ap.parse_args()
    out = Path(args.out).resolve()
    for folder, name in SPLITS.items():
        src = Path(args.root) / f"VisDrone2019-DET-{folder}"
        if not src.exists():
            print(f"bỏ qua {name}: không thấy {src}")
            continue
        print(f"{name}: {convert_split(src, out / name)} hộp")
    names = "\n".join(f"  {i}: {c}" for i, c in enumerate(CLASSES))
    (out / "visdrone.yaml").write_text(
        f"path: {out}\ntrain: train/images\nval: val/images\ntest: test/images\nnames:\n{names}\n")
    print("đã tạo", out / "visdrone.yaml")


if __name__ == "__main__":
    main()
