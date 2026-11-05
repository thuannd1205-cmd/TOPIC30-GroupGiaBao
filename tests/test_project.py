"""Kiểm thử cơ bản các thành phần và cú pháp của dự án."""
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class ProjectStructureTests(unittest.TestCase):
    def test_required_files_exist(self):
        for name in ("app.py", "train.py", "predict.py", "data.yaml",
                     "requirements.txt", "models/best.pt", "classes_vi.json"):
            with self.subTest(name=name):
                self.assertTrue((ROOT / name).exists(), f"Thiếu {name}")

    def test_class_translation_is_json(self):
        data = json.loads((ROOT / "classes_vi.json").read_text(encoding="utf-8"))
        self.assertIsInstance(data, dict)
        self.assertGreater(len(data), 0)

    def test_data_yaml_declares_classes(self):
        import yaml
        data = yaml.safe_load((ROOT / "data.yaml").read_text(encoding="utf-8"))
        self.assertEqual(data.get("nc"), len(data.get("names", [])))
        self.assertEqual(data.get("nc"), 58)


if __name__ == "__main__":
    unittest.main()
