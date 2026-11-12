"""Huấn luyện YOLOv12m / YOLO26m / RT-DETR trên VisDrone (cấu hình theo Bảng 3.5 của báo cáo).

    python src/train.py --model yolo26m
    python src/train.py --model yolo12m --epochs 50 --batch 16
    python src/train.py --model rtdetr
"""
import argparse

from ultralytics import RTDETR, YOLO

# tên rút gọn -> (lớp mô hình, trọng số COCO)
MODELS = {
    "yolo12m": (YOLO, "yolo12m.pt"),
    "yolo26m": (YOLO, "yolo26m.pt"),
    "rtdetr": (RTDETR, "rtdetr-l.pt"),  # báo cáo dùng backbone ResNet18; đổi bằng --weights
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=MODELS, default="yolo26m")
    ap.add_argument("--weights", help="ghi đè trọng số khởi tạo (.pt hoặc .yaml)")
    ap.add_argument("--data", default="datasets/visdrone/visdrone.yaml")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--lr0", type=float, default=1e-4)
    ap.add_argument("--device", default="0")
    ap.add_argument("--project", default="runs/train")
    args = ap.parse_args()

    cls, weights = MODELS[args.model]
    model = cls(args.weights or weights)
    model.train(data=args.data, epochs=args.epochs, batch=args.batch, imgsz=args.imgsz,
                optimizer="AdamW", lr0=args.lr0, device=args.device,
                project=args.project, name=args.model)


if __name__ == "__main__":
    main()
