# Phát hiện đối tượng cho hệ thống giám sát từ UAV (VisDrone2019-DET)

So sánh bốn detector — Faster R-CNN, RT-DETR, YOLOv12m, YOLO26m — trên VisDrone2019-DET,
tối ưu suy luận bằng TensorRT và xây dựng ứng dụng tracking (ByteTrack / BoT-SORT) trong [demo/](demo/).

## Dữ liệu

| Tập | Số ảnh | Số hộp | Vai trò |
|---|---:|---:|---|
| Train | 6.471 | 344.736 | Huấn luyện |
| Validation | 548 | 38.791 | Theo dõi huấn luyện |
| Test-dev | 1.610 | 75.367 | Đánh giá cuối |

- 10 lớp: pedestrian, people, bicycle, car, van, truck, tricycle, awning-tricycle, bus, motor.
- 62,2% đối tượng là small (trung vị diện tích ~629 pixel²); car, pedestrian chiếm ưu thế, tricycle / awning-tricycle / bus hiếm.
- Bỏ vùng ignored (0) và others (11), loại hộp lỗi, cắt hộp vượt biên.

## Cài đặt và chạy

```bash
pip install -r requirements.txt

# 1. Chuyển VisDrone sang định dạng YOLO (đặt các thư mục VisDrone2019-DET-* ở thư mục gốc)
python src/prepare_data.py --root . --out datasets/visdrone

# 2. Huấn luyện: yolo12m | yolo26m | rtdetr
python src/train.py --model yolo26m

# 3. Đánh giá trên test-dev
python src/test.py --weights runs/train/yolo26m/weights/best.pt
python src/test.py --weights runs/train/rtdetr/weights/best.pt --rtdetr

# 4. (tuỳ chọn) Xuất TensorRT rồi đánh giá lại
python src/export_trt.py --weights runs/train/yolo26m/weights/best.pt --precision fp16
python src/test.py --weights runs/train/yolo26m/weights/best.engine
```

Cấu hình huấn luyện mặc định: 50 epoch, batch 16, ảnh 640 × 640, AdamW, lr 1e-4, khởi tạo từ trọng số COCO.
Báo cáo dùng RT-DETR backbone ResNet18; `--model rtdetr` mặc định dùng `rtdetr-l.pt` của Ultralytics,
đổi bằng `--weights`. Faster R-CNN (ResNet-50-FPN) được huấn luyện riêng bằng torchvision, không có trong `src/`.

## Kết quả

GPU NVIDIA T4 (16 GB), Ultralytics + PyTorch, mỗi mô hình huấn luyện một lần, đánh giá trên 1.610 ảnh test-dev.

### Độ chính xác

| Mô hình | mAP50 | mAP50:95 | Params (M) | GFLOPs |
|---|---:|---:|---:|---:|
| Faster R-CNN | 32,9% | 19,4% | 41,4 | 208 |
| RT-DETR | 37,0% | 21,3% | 20,0 | 61,2 |
| YOLOv12m | 36,5% | 21,1% | 20,1 | 70,4 |
| **YOLO26m** | **37,9%** | **21,7%** | 21,9 | 75,4 |

### Tốc độ suy luận (T4)

| Mô hình | FPS PyTorch | FPS TensorRT | Tăng tốc |
|---|---:|---:|---:|
| Faster R-CNN | 20 | 38 | 1,90× |
| RT-DETR | 51 | 198 | 3,88× |
| YOLOv12m | 48 | 143 | 2,98× |
| **YOLO26m** | **53** | **208** | **3,92×** |

### YOLO26m theo mức độ chính xác số học TensorRT

| Precision | mAP50 | mAP50:95 | ms/ảnh | FPS | Engine (MiB) |
|---|---:|---:|---:|---:|---:|
| FP32 | 0,379 | 0,217 | 19,4 | 51,5 | 254 |
| FP16 | 0,378 | 0,216 | 4,8 | 208,3 | 182 |
| INT8 | 0,373 | 0,211 | 4,3 | 232,6 | 60 |

### Nhận xét

- YOLO26m cao nhất ở cả mAP50 và mAP50:95, và cũng nhanh nhất ở cả hai chế độ (PyTorch, TensorRT).
  Với ít hơn ~47% tham số và ~64% GFLOPs so với Faster R-CNN, nó vẫn cao hơn 5,0 / 2,3 điểm phần trăm (mAP50 / mAP50:95).
- Khác biệt giữa các mô hình tập trung ở đối tượng nhỏ; vẫn còn nhầm lớp (car/van, pedestrian/people) và hộp chồng lấn.
- FP16 giảm thời gian inference ~4× với mAP giảm 0,1 điểm phần trăm; INT8 giảm engine còn 60 MiB nhưng mAP giảm 0,6 điểm phần trăm.
- Hạn chế: mỗi mô hình chỉ huấn luyện một lần (chưa có nhiều seed); chưa có AP theo lớp / kích thước đầy đủ;
  chưa có MOTA, IDF1, HOTA cho hai bộ tracker; augmentation chọn lọc và class_weight mới ở mức đề xuất, chưa có ablation.

## Cấu trúc

```
src/prepare_data.py   chuyển VisDrone -> YOLO, tạo visdrone.yaml
src/train.py          huấn luyện YOLOv12m / YOLO26m / RT-DETR
src/test.py           đánh giá trên test-dev
src/export_trt.py     xuất TensorRT FP32/FP16/INT8
demo/                 ứng dụng tracking (Streamlit + FastAPI trên Colab)
```

Dataset, báo cáo, slide và trọng số đã được loại khỏi git qua [.gitignore](.gitignore).
