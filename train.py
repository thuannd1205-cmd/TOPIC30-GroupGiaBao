"""
Huấn luyện YOLO trên dataset Vietnam-Traffic-Sign-Detection (Roboflow, định dạng YOLOv8).

Ví dụ:
  python train.py --dataset ./Vietnam-Traffic-Sign-Detection_v6i_yolov8 --model yolov8s.pt --epochs 100 --device 0

--dataset là thư mục đã giải nén, chứa data.yaml, train/, valid/, test/.
Kết thúc, trọng số tốt nhất được chép vào models/best.pt để app.py sử dụng.
"""
import argparse
import shutil
from pathlib import Path

import yaml
from ultralytics import YOLO


def build_data_yaml(dataset: Path) -> Path:
    """Tạo data.yaml với đường dẫn tuyệt đối (tránh lỗi '../train/images' của Roboflow)."""
    src = yaml.safe_load((dataset / "data.yaml").read_text(encoding="utf-8"))
    cfg = {"path": str(dataset.resolve()), "train": "train/images", "val": "valid/images",
           "names": src["names"]}
    if (dataset / "test" / "images").exists():
        cfg["test"] = "test/images"
    out = Path("runs"); out.mkdir(exist_ok=True)
    p = out / "data_vn_traffic_sign.yaml"
    p.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="Thư mục dataset đã giải nén")
    ap.add_argument("--model", default="yolov8s.pt", help="yolov8n/s/m.pt hoặc yolo11n/s/m.pt")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=960, help="Biển báo nhỏ nên dùng 960 hoặc 1280")
    ap.add_argument("--batch", type=int, default=8, help="Giảm nếu hết VRAM (có thể dùng -1 để tự chọn)")
    ap.add_argument("--device", default=None, help="0 cho GPU, cpu cho CPU")
    ap.add_argument("--name", default="vn_traffic_sign")
    a = ap.parse_args()

    data = build_data_yaml(Path(a.dataset))
    model = YOLO(a.model)
    model.train(data=str(data), epochs=a.epochs, imgsz=a.imgsz, batch=a.batch,
                device=a.device, project="runs", name=a.name, patience=30,
                fliplr=0.0,   # KHÔNG lật ngang: biển rẽ trái/phải (P.123a/b, W.201a/b...) sẽ bị nhầm nhãn
                mosaic=1.0, close_mosaic=10)

    m = model.val(split="test") if (Path(a.dataset) / "test" / "images").exists() else model.val()
    print(f"mAP50: {m.box.map50:.3f} | mAP50-95: {m.box.map:.3f}")

    best = Path(model.trainer.save_dir) / "weights" / "best.pt"
    Path("models").mkdir(exist_ok=True)
    shutil.copy(best, "models/best.pt")
    print("Đã chép trọng số tốt nhất vào models/best.pt")


if __name__ == "__main__":
    main()
