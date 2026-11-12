"""Đánh giá mô hình trên VisDrone2019-DET test-dev (mAP50, mAP50:95, tốc độ).

    python src/test.py --weights runs/train/yolo26m/weights/best.pt
    python src/test.py --weights best.engine      # engine TensorRT
"""
import argparse

from ultralytics import RTDETR, YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--rtdetr", action="store_true", help="dùng khi weights là RT-DETR")
    ap.add_argument("--data", default="datasets/visdrone/visdrone.yaml")
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--device", default="0")
    args = ap.parse_args()

    model = (RTDETR if args.rtdetr else YOLO)(args.weights)
    m = model.val(data=args.data, split=args.split, imgsz=args.imgsz,
                  batch=args.batch, device=args.device)
    print(f"mAP50    : {m.box.map50:.3f}")
    print(f"mAP50:95 : {m.box.map:.3f}")
    print(f"Tốc độ (ms/ảnh): {m.speed}")


if __name__ == "__main__":
    main()
