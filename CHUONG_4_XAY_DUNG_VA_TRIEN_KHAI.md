# CHƯƠNG 4. XÂY DỰNG VÀ TRIỂN KHAI HỆ THỐNG

> Tài liệu này mô tả đúng các thành phần có trong repository. Các thông số thực nghiệm phụ thuộc máy tính và lần huấn luyện thực tế; cần điền kết quả đo được, không tự tạo số liệu.

## 4.1. Xây dựng môi trường lập trình

Hệ thống được xây dựng bằng Python, sử dụng Ultralytics để triển khai mô hình YOLO, OpenCV để đọc và xử lý ảnh/video, và Streamlit để xây dựng giao diện. Dự án khuyến nghị Python 3.10–3.12.

Tại thư mục gốc của dự án, tạo môi trường ảo và cài đặt thư viện:

```powershell
py -3.10 -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Các thư viện chính gồm `ultralytics`, `opencv-python-headless`, `streamlit`, `numpy`, `pandas`, `Pillow`, `PyYAML` và `imageio-ffmpeg`. Môi trường ảo giúp cô lập thư viện của dự án với các chương trình Python khác.

Chạy `python scripts/check_environment.py` để xem phiên bản Python, Ultralytics, OpenCV, PyTorch và trạng thái CUDA. Nếu `torch.cuda.is_available()` trả về `True`, chương trình có thể sử dụng GPU NVIDIA tương thích; nếu không, mô hình vẫn có thể chạy trên CPU. Huấn luyện bằng CPU thường chậm hơn GPU.

## 4.2. Thu thập và chuẩn bị bộ dữ liệu

Dự án sử dụng bộ **Vietnam-Traffic-Sign-Detection**, phiên bản 6, được chia sẻ trên Roboflow Universe. Thông tin đi kèm dataset cho biết bộ dữ liệu có 3.680 ảnh, 58 lớp và nhãn theo định dạng YOLOv8. Ảnh được resize về 640×640 và áp dụng auto-contrast. Nguồn: https://universe.roboflow.com/vietnam-traffic-sign-detection/vietnam-traffic-sign-detection-2i2j8

Tải bộ dữ liệu ở định dạng YOLOv8, giải nén và tổ chức theo cấu trúc:

```text
dataset/
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

Tệp `data.yaml` khai báo đường dẫn các tập dữ liệu, số lớp `nc = 58` và danh sách tên lớp. Mỗi ảnh có thể có tệp nhãn `.txt` cùng tên. Một dòng nhãn YOLO gồm 5 giá trị: `class_id x_center y_center width height`; các tọa độ được chuẩn hóa trong khoảng từ 0 đến 1.

Dùng lệnh sau để thống kê dữ liệu thực tế và rà soát lỗi nhãn:

```powershell
python scripts/analyze_dataset.py --dataset datasets/Vietnam-Traffic-Sign-Detection
```

Bảng số lượng ảnh train/validation/test, số nhãn và số đối tượng theo lớp cần lấy từ kết quả script `runs/dataset_report.json` sau khi chạy trên đúng bộ dữ liệu đã tải. Không suy ra số ảnh mỗi tập chỉ từ tổng số ảnh.

## 4.3. Cấu hình và huấn luyện mô hình YOLO

Dự án sử dụng thư viện Ultralytics YOLO. Tệp `train.py` cho phép lựa chọn trọng số khởi tạo, số epoch, kích thước ảnh, batch size và thiết bị huấn luyện. Ví dụ:

```powershell
python train.py --dataset datasets/Vietnam-Traffic-Sign-Detection --model yolov8s.pt --epochs 100 --imgsz 960 --batch 8 --device 0
```

Trong đó `yolov8s.pt` là mô hình YOLOv8 kích thước small; `epochs` là số vòng huấn luyện; `imgsz` là kích thước ảnh đầu vào; `batch` là số ảnh trong một batch; `device 0` chỉ GPU thứ nhất. Có thể thay bằng `--device cpu` nếu không có GPU CUDA.

Trong quá trình huấn luyện, Ultralytics lưu log, biểu đồ và trọng số trong thư mục `runs/`. Tệp `best.pt` là trọng số có kết quả validation tốt nhất theo tiêu chí của quá trình huấn luyện. Sau khi huấn luyện, `train.py` sao chép trọng số này vào `models/best.pt` để ứng dụng sử dụng.

Khi viết báo cáo thực nghiệm, ghi lại model, epochs, imgsz, batch, thiết bị, thời gian huấn luyện và các chỉ số Precision, Recall, mAP50, mAP50-95 từ kết quả chạy thật. Không ghi các giá trị chưa được đo.

## 4.4. Xây dựng chức năng phát hiện và nhận diện biển báo từ hình ảnh

Chức năng xử lý ảnh được tích hợp trong `app.py`, mục **Nhận diện ảnh**. Người dùng tải một hoặc nhiều ảnh JPG, JPEG, PNG, BMP hoặc WEBP. Ảnh được đọc bằng Pillow, chuyển về mảng ảnh và đưa qua mô hình YOLO bằng `model.predict()`.

Kết quả dự đoán chứa hộp giới hạn, mã lớp và độ tin cậy. Hệ thống sử dụng `result.plot()` để vẽ khung bao quanh đối tượng. Mã lớp được đối chiếu với `classes_vi.json` để hiển thị tên biển báo tiếng Việt. Ảnh đã nhận diện có thể tải về dưới định dạng PNG; danh sách phát hiện có thể xuất thành CSV.

## 4.5. Xây dựng chức năng nhận diện biển báo từ video

Trong mục **Nhận diện video**, người dùng tải video MP4, AVI, MOV hoặc MKV. OpenCV mở video bằng `cv2.VideoCapture` và đọc lần lượt từng khung hình. Mỗi khung hình được đưa vào `model.predict()` hoặc `model.track()` nếu bật chức năng tracking.

Kết quả được vẽ lên từng khung hình và ghi thành video bằng `cv2.VideoWriter`. Giao diện hiển thị tiến trình xử lý, khung hình xem trước và tốc độ xử lý. Video đầu ra được mã hóa H.264 khi công cụ FFmpeg khả dụng, sau đó cho phép tải về cùng bảng thống kê CSV.

## 4.6. Xây dựng giao diện chương trình

Giao diện sử dụng Streamlit, được triển khai trong `app.py`. Thanh bên cho phép chọn trọng số có sẵn `models/best.pt` hoặc tải tệp `.pt`, điều chỉnh confidence, IoU và kích thước ảnh đầu vào. Đối với video, người dùng có thể bật/tắt tracking và thay đổi tần suất xem trước.

Giao diện chính gồm hai tab: **Nhận diện ảnh** và **Nhận diện video**. Kết quả được trình bày trực quan, kèm thông tin số lượng biển báo và chức năng tải tệp kết quả. Khởi chạy giao diện bằng:

```powershell
streamlit run app.py
```

## 4.7. Tích hợp và kiểm thử hệ thống

Các thành phần được tích hợp theo luồng: chọn trọng số → tải ảnh/video → tiền xử lý → dự đoán bằng YOLO → vẽ kết quả → hiển thị và xuất tệp. Có thể kiểm tra cấu trúc và một số thành phần cơ bản bằng:

```powershell
python -m unittest discover -s tests -v
python -m compileall app.py train.py predict.py scripts tests
```

Các trường hợp kiểm thử đề xuất:
- Khởi động ứng dụng khi có trọng số hợp lệ.
- Tải ảnh đúng định dạng và ảnh không hỗ trợ.
- Tải video hợp lệ và video không đọc được.
- Thử thay đổi ngưỡng confidence và IoU.
- Kiểm tra trường hợp không phát hiện được biển báo.
- Tải ảnh kết quả, video kết quả và tệp CSV.
- Thử chạy trên CPU; nếu có GPU NVIDIA thì kiểm tra thêm CUDA.

Khi hoàn thiện báo cáo, bổ sung ảnh chụp giao diện, kết quả nhận diện ảnh/video, log kiểm thử và thông tin thiết bị thực tế. Không khẳng định một trường hợp đã kiểm thử thành công nếu chưa chạy thử.

## 4.8. Kết luận chương 4

Chương 4 đã trình bày quá trình xây dựng môi trường, chuẩn bị dữ liệu, cấu hình và huấn luyện mô hình YOLO, phát hiện biển báo từ ảnh và video, xây dựng giao diện Streamlit và kiểm thử hệ thống. Dự án tổ chức các chức năng thành những tệp riêng như `train.py`, `predict.py`, `app.py` và các công cụ hỗ trợ trong thư mục `scripts/`. Trọng số `models/best.pt` được sử dụng để triển khai nhận diện. Kết quả định lượng cuối cùng cần được cập nhật dựa trên lần huấn luyện và kiểm thử thực tế của nhóm.
