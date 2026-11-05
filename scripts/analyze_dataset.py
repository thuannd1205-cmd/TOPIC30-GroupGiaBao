"""Thống kê dataset YOLO: số ảnh, nhãn, đối tượng mỗi lớp và lỗi nhãn."""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def read_class_names(data_yaml):
    try:
        import yaml
        data = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
        names = data.get("names", [])
        if isinstance(names, dict):
            return {int(k): str(v) for k, v in names.items()}
        return {i: str(name) for i, name in enumerate(names)}
    except Exception:
        return {}


def inspect_split(root, split, names):
    image_dir = root / split / "images"
    label_dir = root / split / "labels"
    images = sorted(p for p in image_dir.glob("*") if p.suffix.lower() in IMAGE_EXTS) if image_dir.exists() else []
    labels = sorted(label_dir.glob("*.txt")) if label_dir.exists() else []
    counts = Counter()
    errors = []
    missing_labels = 0

    for image in images:
        label = label_dir / (image.stem + ".txt")
        if not label.exists():
            missing_labels += 1
            continue
        for line_no, line in enumerate(label.read_text(encoding="utf-8").splitlines(), 1):
            parts = line.split()
            if not parts:
                continue
            if len(parts) != 5:
                errors.append(f"{label}:{line_no}: cần 5 giá trị, nhận {len(parts)}")
                continue
            try:
                class_id = int(float(parts[0]))
                coords = [float(x) for x in parts[1:]]
                if class_id < 0 or (names and class_id not in names):
                    raise ValueError("class_id không hợp lệ")
                if any(x < 0 or x > 1 for x in coords):
                    raise ValueError("tọa độ YOLO phải nằm trong [0, 1]")
                counts[class_id] += 1
            except ValueError as exc:
                errors.append(f"{label}:{line_no}: {exc}")

    return {
        "split": split, "images": len(images), "label_files": len(labels),
        "images_without_label": missing_labels, "objects": sum(counts.values()),
        "class_counts": counts, "errors": errors,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="Thư mục chứa data.yaml, train/, valid/, test/")
    args = parser.parse_args()
    root = Path(args.dataset).resolve()
    yaml_path = root / "data.yaml"
    if not yaml_path.exists():
        raise SystemExit(f"Không tìm thấy {yaml_path}")

    names = read_class_names(yaml_path)
    splits = [s for s in ("train", "valid", "test") if (root / s).exists()]
    if not splits:
        raise SystemExit("Không tìm thấy thư mục train, valid hoặc test.")

    report = {"dataset": str(root), "class_count_from_yaml": len(names), "splits": []}
    total_counts = Counter()
    for split in splits:
        result = inspect_split(root, split, names)
        split_counts = result.pop("class_counts")
        total_counts.update(split_counts)
        report["splits"].append({
            **result,
            "class_counts": {
                str(cid): {"name": names.get(cid, f"class_{cid}"), "objects": count}
                for cid, count in sorted(split_counts.items())
            }
        })
        print(f"{split:>5}: {result['images']} ảnh | {result['label_files']} file nhãn | "
              f"{result['objects']} đối tượng | thiếu nhãn: {result['images_without_label']} | "
              f"lỗi: {len(result['errors'])}")
        for error in result["errors"][:10]:
            print("  -", error)

    report["total_images"] = sum(x["images"] for x in report["splits"])
    report["total_label_files"] = sum(x["label_files"] for x in report["splits"])
    report["total_objects"] = sum(x["objects"] for x in report["splits"])
    report["class_counts"] = {
        str(cid): {"name": names.get(cid, f"class_{cid}"), "objects": count}
        for cid, count in sorted(total_counts.items())
    }

    out = Path("runs")
    out.mkdir(exist_ok=True)
    (out / "dataset_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    with (out / "dataset_class_counts.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["class_id", "class_name", "object_count"])
        for cid, count in sorted(total_counts.items()):
            writer.writerow([cid, names.get(cid, f"class_{cid}"), count])
    print(f"\nTổng: {report['total_images']} ảnh, {report['total_objects']} đối tượng.")
    print("Đã lưu báo cáo tại runs/dataset_report.json và runs/dataset_class_counts.csv")


if __name__ == "__main__":
    main()
