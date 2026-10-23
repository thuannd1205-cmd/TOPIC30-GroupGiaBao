"""
Kiểm tra nhanh mô hình bằng dòng lệnh (không cần giao diện).

Ví dụ:
  python predict.py --source anh_test.jpg
  python predict.py --source thu_muc_anh/ --conf 0.4
  python predict.py --source video.mp4
Kết quả (ảnh/video đã vẽ box) lưu trong thư mục runs/predict/
"""
import argparse
import json
from collections import Counter
from pathlib import Path

from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="Ảnh, video hoặc thư mục ảnh")
    ap.add_argument("--weights", default="models/best.pt")
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--iou", type=float, default=0.45)
    ap.add_argument("--imgsz", type=int, default=960, help="Nên bằng imgsz lúc train")
    a = ap.parse_args()

    vi_file = Path(__file__).with_name("classes_vi.json")
    vi = json.loads(vi_file.read_text(encoding="utf-8")) if vi_file.exists() else {}

    model = YOLO(a.weights)
    counter = Counter()
    for r in model.predict(a.source, conf=a.conf, iou=a.iou, imgsz=a.imgsz, save=True,
                           project="runs", name="predict", exist_ok=True, stream=True, verbose=False):
        print(f"\n{Path(r.path).name}: {len(r.boxes)} biển báo")
        for b in r.boxes:
            code = r.names[int(b.cls[0])]
            counter[code] += 1
            print(f"  - {code:8s} {vi.get(code, ''):45s} conf={float(b.conf[0]):.2f}")

    print("\n===== THỐNG KÊ =====")
    for code, n in counter.most_common():
        print(f"{code:8s} {vi.get(code, ''):45s} {n}")
    print(f"Tổng: {sum(counter.values())} biển báo. Kết quả đã lưu trong runs/predict/")


if __name__ == "__main__":
    main()
