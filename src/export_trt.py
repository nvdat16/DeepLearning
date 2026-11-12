"""Xuất mô hình sang TensorRT (FP32 / FP16 / INT8).

    python src/export_trt.py --weights best.pt --precision fp16
"""
import argparse

from ultralytics import RTDETR, YOLO

ap = argparse.ArgumentParser()
ap.add_argument("--weights", required=True)
ap.add_argument("--rtdetr", action="store_true")
ap.add_argument("--precision", choices=["fp32", "fp16", "int8"], default="fp16")
ap.add_argument("--data", default="datasets/visdrone/visdrone.yaml", help="dữ liệu hiệu chuẩn INT8")
ap.add_argument("--imgsz", type=int, default=640)
args = ap.parse_args()

model = (RTDETR if args.rtdetr else YOLO)(args.weights)
model.export(format="engine", half=args.precision == "fp16", int8=args.precision == "int8",
             data=args.data, imgsz=args.imgsz, batch=1)
