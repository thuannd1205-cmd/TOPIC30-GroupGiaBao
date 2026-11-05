"""Kiểm tra phiên bản Python, thư viện và khả năng sử dụng GPU."""
import sys

print("=" * 60)
print("KIỂM TRA MÔI TRƯỜNG DỰ ÁN YOLO")
print("=" * 60)
print("Python:", sys.version.split()[0])

try:
    import torch
    print("PyTorch:", torch.__version__)
    print("CUDA được PyTorch nhận diện:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
    else:
        print("Thiết bị chạy mô hình: CPU (hoặc cài bản PyTorch CUDA nếu có GPU NVIDIA)")
except ImportError:
    print("Chưa cài PyTorch. Hãy chạy: pip install -r requirements.txt")

try:
    import ultralytics
    print("Ultralytics:", ultralytics.__version__)
except ImportError:
    print("Thiếu ultralytics")

try:
    import cv2
    print("OpenCV:", cv2.__version__)
except ImportError:
    print("Thiếu OpenCV")

try:
    import streamlit
    print("Streamlit:", streamlit.__version__)
except ImportError:
    print("Thiếu Streamlit")

print("=" * 60)
