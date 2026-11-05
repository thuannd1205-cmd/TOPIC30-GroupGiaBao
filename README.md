# TOPIC30 – Hệ thống phát hiện và nhận diện biển báo giao thông bằng YOLO

## 1. Giới thiệu
Dự án sử dụng Ultralytics YOLO để phát hiện biển báo giao thông Việt Nam từ hình ảnh và video. Giao diện được xây dựng bằng Streamlit.

Các chức năng chính:
- Kiểm tra môi trường Python, thư viện và CPU/GPU.
- Thống kê bộ dữ liệu YOLO (ảnh, nhãn, số lớp, lỗi nhãn).
- Huấn luyện mô hình bằng `train.py`.
- Nhận diện hình ảnh và video bằng giao diện `app.py`.
- Tải ảnh/video kết quả và bảng thống kê CSV.

## 2. Cấu trúc thư mục

```text
TOPIC30-GroupGiaBao/
├── app.py
├── train.py
├── predict.py
├── data.yaml
├── classes_vi.json
├── requirements.txt
├── models/
│   └── best.pt
├── scripts/
│   ├── check_environment.py
│   └── analyze_dataset.py
├── tests/
│   └── test_project.py
├── CHUONG_4_XAY_DUNG_VA_TRIEN_KHAI.md
└── README.md
```

Dataset tải riêng, không đưa lên GitHub vì dung lượng lớn. Sau khi giải nén, thư mục dataset cần có dạng:

```text
Vietnam-Traffic-Sign-Detection/
├── data.yaml
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

## 3. Cài đặt trên Windows

Mở Terminal tại thư mục dự án (thư mục có file `requirements.txt`), sau đó chạy lần lượt:

```powershell
py -3.10 -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Nếu PowerShell chặn kích hoạt môi trường ảo, chạy lệnh sau trong cửa sổ Terminal hiện tại rồi kích hoạt lại:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv\Scripts\Activate.ps1
```

Kiểm tra môi trường:

```powershell
python scripts/check_environment.py
```

## 4. Chuẩn bị và thống kê dữ liệu

Tải dataset Vietnam-Traffic-Sign-Detection phiên bản 6 tại:
https://universe.roboflow.com/vietnam-traffic-sign-detection/vietnam-traffic-sign-detection-2i2j8

Chọn định dạng YOLOv8, giải nén dataset. Ví dụ dataset nằm tại `datasets/Vietnam-Traffic-Sign-Detection`, chạy:

```powershell
python scripts/analyze_dataset.py --dataset datasets/Vietnam-Traffic-Sign-Detection
```

Script sẽ thống kê số ảnh, số file nhãn, số đối tượng theo lớp và các lỗi định dạng nhãn. Kết quả được lưu ở `runs/dataset_report.json` và `runs/dataset_class_counts.csv`.

## 5. Huấn luyện mô hình

Ví dụ huấn luyện YOLOv8s trong 100 epochs:

```powershell
python train.py --dataset datasets/Vietnam-Traffic-Sign-Detection --model yolov8s.pt --epochs 100 --imgsz 960 --batch 8 --device 0
```

- `--device 0`: dùng GPU NVIDIA CUDA.
- `--device cpu`: dùng CPU (thường chậm hơn).
- Có thể bỏ `--device` để Ultralytics tự chọn thiết bị.
- Nếu hết bộ nhớ GPU, giảm `--batch` hoặc `--imgsz`.

Kết quả huấn luyện nằm trong `runs/`; trọng số tốt nhất được sao chép vào `models/best.pt`. Không ghi kết quả huấn luyện giả định vào báo cáo; lấy số liệu thực tế từ `runs/detect/.../results.csv` và biểu đồ trong thư mục kết quả.

## 6. Chạy ứng dụng

```powershell
streamlit run app.py
```

Trình duyệt sẽ mở giao diện. Chọn trọng số ở thanh bên, điều chỉnh confidence/IoU/kích thước ảnh, sau đó:
- Tab **Nhận diện ảnh**: tải một hoặc nhiều ảnh; xem ảnh đã vẽ khung, mã biển, độ tin cậy; tải ảnh kết quả và CSV.
- Tab **Nhận diện video**: tải video, nhấn **Bắt đầu nhận diện video**; xem tiến trình và video kết quả; tải video và CSV.
- Có thể bật tracking để theo dõi ID đối tượng qua các khung hình.

Có thể chạy dự đoán bằng dòng lệnh:

```powershell
python predict.py --source anh_thu.jpg
python predict.py --source video_thu.mp4
```

Kết quả dự đoán CLI được lưu trong `runs/predict/`.

## 7. Kiểm thử

```powershell
python -m unittest discover -s tests -v
```

Kiểm thử cú pháp:

```powershell
python -m compileall app.py train.py predict.py scripts tests
```

## 8. Đưa dự án lên GitHub

Trong Terminal tại thư mục dự án:

```powershell
git status
git add .
git commit -m "Hoan thien chuong 4 - xay dung va trien khai he thong"
git push
```

Nếu chưa clone repository, clone trước:

```powershell
git clone https://github.com/thuannd1205-cmd/TOPIC30-GroupGiaBao.git
```

Sau đó chép các file đã cập nhật vào thư mục vừa clone, rồi chạy các lệnh `git add`, `git commit`, `git push` ở trên. Không đưa `venv/`, dataset hoặc thư mục `runs/` lên GitHub.
