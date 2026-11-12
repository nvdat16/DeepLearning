"""Thống kê bounding box của 10 class VisDrone2019-DET, theo từng tập.

    .venv/bin/python class_stats.py .
    .venv/bin/python class_stats.py /path/to/data --out-dir presentation/dataset_stats

Số liệu chính bỏ category 0/11, score <= 0, width/height <= 0. CSV raw giữ
toàn bộ nhãn category 1–10 để đối chiếu. Đơn vị là số bounding box, không phải số ảnh.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager, ticker
import numpy as np

CLASSES = ["pedestrian", "people", "bicycle", "car", "van", "truck",
           "tricycle", "awning-tricycle", "bus", "motor"]
SPLITS = ["train", "val", "test-dev"]
OUT = Path(__file__).resolve().parent / "presentation" / "dataset_stats"


def analyze_split(root, split):
    folder = Path(root) / f"VisDrone2019-DET-{split}"
    files = sorted((folder / "annotations").glob("*.txt"))
    if not files:
        raise FileNotFoundError(f"Không tìm thấy nhãn: {folder / 'annotations'}")
    counts, raw, presence = (np.zeros(10, dtype=np.int64) for _ in range(3))
    categories = np.zeros(12, dtype=np.int64)
    wh, invalid = [], []
    score_zero = 0
    digest = hashlib.sha256()
    for path in files:
        content = path.read_bytes()
        digest.update(path.name.encode() + b"\0" + content + b"\0")
        seen = set()
        for line, row in enumerate(csv.reader(content.decode("utf-8-sig").splitlines()), 1):
            if not row or not any(x.strip() for x in row):
                continue
            if len(row) < 8:
                raise ValueError(f"Thiếu cột nhãn: {path}:{line}")
            a = np.asarray([float(x.strip()) for x in row[:8]])
            if not np.isfinite(a).all() or a[5] != int(a[5]) or not 0 <= a[5] <= 11:
                raise ValueError(f"Nhãn/category không hợp lệ: {path}:{line}")
            cat = int(a[5]); categories[cat] += 1
            target = 1 <= cat <= 10
            bad_size = a[2] <= 0 or a[3] <= 0
            if target:
                raw[cat-1] += 1
            if bad_size:
                invalid.append(dict(file=path.name, line=line, category=cat, width=float(a[2]), height=float(a[3])))
            if target and a[4] <= 0:
                score_zero += 1
            if target and not bad_size and a[4] > 0:
                counts[cat-1] += 1; seen.add(cat-1); wh.append(a[2:4])
        for cat in seen:
            presence[cat] += 1
    images = {p.stem for p in (folder / "images").glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}}
    if images and images != {p.stem for p in files}:
        raise ValueError(f"Ảnh và nhãn không khớp: {folder}")
    audit = dict(images=len(images), annotations=len(files), raw_rows=int(categories.sum()),
        raw_target_boxes=int(raw.sum()), valid_target_boxes=int(counts.sum()),
        ignored_regions=int(categories[0]), others=int(categories[11]), target_score_zero=score_zero,
        invalid_boxes=invalid, annotations_sha256=digest.hexdigest())
    return dict(counts=counts, raw=raw, presence=presence,
                wh=np.asarray(wh, dtype=float).reshape(-1, 2), audit=audit)


def count_classes(root, split):
    return analyze_split(root, split)["counts"]


def plot_distribution(counts, title, output):
    order = np.argsort(-counts, kind="stable")
    values = counts[order]
    fig, ax = plt.subplots(figsize=(13.5, 5.7), facecolor="white")
    colors = ["#4472C4" if i not in [6, 7, 8] else "#ED7D31" for i in order]
    bars = ax.barh(np.arange(10), values, height=.68, color=colors)
    ax.invert_yaxis()
    ax.set_yticks(np.arange(10), [CLASSES[i] for i in order], fontsize=17)
    ax.set_xlim(0, max(values) * 1.37)
    ax.set_xlabel("Số bounding box", fontsize=17, labelpad=9)
    ax.set_title(title, fontsize=22, loc="left", pad=15, weight="bold")
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:,.0f}".replace(",", ".")))
    ax.tick_params(axis="x", labelsize=13)
    ax.tick_params(axis="y", length=0)
    ax.spines[["top", "right", "left"]].set_visible(False)
    for bar, n in zip(bars, values):
        label = f"{n:,}".replace(",", ".") + f"  ({100*n/counts.sum():.1f}%)".replace(".", ",")
        ax.text(n + max(values)*.016, bar.get_y()+bar.get_height()/2, label, va="center", fontsize=16)
    ax.set_axisbelow(True); ax.grid(axis="x", color="#E8E8E8", linewidth=.7)
    fig.tight_layout()
    fig.savefig(output.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(output.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--out-dir", type=Path, default=OUT)
    args = parser.parse_args(); args.out_dir.mkdir(parents=True, exist_ok=True)
    font = Path("/System/Library/Fonts/Supplemental/Arial.ttf")
    if font.is_file():
        font_manager.fontManager.addfont(str(font)); plt.rcParams["font.family"] = "Arial"
    data = {s: analyze_split(args.root, s) for s in SPLITS}
    counts = {s: data[s]["counts"] for s in SPLITS}
    total = np.sum(list(counts.values()), axis=0)
    for name, key in [("class_counts.csv", "counts"), ("raw_class_counts.csv", "raw")]:
        values = {s:data[s][key] for s in SPLITS}; all_counts=np.sum(list(values.values()), axis=0)
        with (args.out_dir/name).open("w", encoding="utf-8-sig", newline="") as f:
            writer=csv.writer(f)
            writer.writerow(["class_id", "class_name", *SPLITS, "total", "share_train_pct", "share_total_pct"])
            for i, cls in enumerate(CLASSES):
                writer.writerow([i+1, cls, *(int(values[s][i]) for s in SPLITS), int(all_counts[i]),
                    round(100*values['train'][i]/values['train'].sum(), 6), round(100*all_counts[i]/all_counts.sum(), 6)])
    with (args.out_dir/"images_with_class.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer=csv.writer(f); writer.writerow(["class_id", "class_name", *SPLITS])
        for i, cls in enumerate(CLASSES):
            writer.writerow([i+1, cls, *(int(data[s]['presence'][i]) for s in SPLITS)])
    summary=dict(definition="category 1..10, score > 0, width > 0, height > 0; đơn vị: bounding box",
        root=str(args.root.resolve()), splits={s:data[s]['audit'] for s in SPLITS},
        total_valid_target_boxes=int(total.sum()), sizes={})
    for s in [*SPLITS, "all"]:
        c = total if s=="all" else counts[s]
        wh=np.concatenate([data[k]['wh'] for k in SPLITS]) if s=="all" else data[s]['wh']
        size_counts=np.histogram(np.prod(wh, axis=1), [0,32**2,96**2,np.inf])[0]
        summary['sizes'][s]=dict(counts=size_counts.tolist(), percent=(100*size_counts/len(wh)).tolist())
        plot_distribution(c, "VisDrone DET • "+("Tất cả các tập" if s=="all" else s)+
            f" • {c.sum():,} hộp".replace(",", "."), args.out_dir/f"class_dist_{s.replace('-', '_')}")
    (args.out_dir/"summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    np.savez_compressed(args.out_dir/"bbox_wh.npz", **{s:data[s]['wh'] for s in SPLITS})
    lines=["SỐ BOUNDING BOX THEO CLASS — VISDRONE2019-DET", summary['definition'], "",
           f"{'class':18s}{'train':>12s}{'val':>12s}{'test-dev':>12s}{'total':>12s}{'train %':>10s}"]
    for i, cls in enumerate(CLASSES):
        lines.append(f"{cls:18s}"+"".join(f"{counts[s][i]:>12,d}" for s in SPLITS)+
                     f"{total[i]:>12,d}{100*counts['train'][i]/counts['train'].sum():>9.2f}%")
    lines += ["TOTAL             "+"".join(f"{counts[s].sum():>12,d}" for s in SPLITS)+f"{total.sum():>12,d}", "",
        "Bỏ category=0 (ignored), category=11 (others), score<=0 và hộp width/height<=0.",
        "Tham khảo summary.json để kiểm tra các hộp bị loại. raw_class_counts.csv giữ hộp lỗi kích thước.",
        "class_weight chỉ tính từ nhãn train; không dùng val/test-dev."]
    text="\n".join(lines); (args.out_dir/"class_counts.txt").write_text(text+"\n", encoding="utf-8")
    print(text); print("\nĐã lưu:",args.out_dir.resolve())


if __name__ == "__main__":
    main()
