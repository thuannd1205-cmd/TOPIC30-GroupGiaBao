"""
HỆ THỐNG PHÁT HIỆN VÀ NHẬN DIỆN BIỂN BÁO GIAO THÔNG BẰNG YOLO
Chạy:  streamlit run app.py
"""
import json
import os
import shutil
import subprocess
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from ultralytics import YOLO

DEFAULT_WEIGHTS = Path("models/best.pt")
_VI_FILE = Path(__file__).with_name("classes_vi.json")
VI_NAMES = json.loads(_VI_FILE.read_text(encoding="utf-8")) if _VI_FILE.exists() else {}

st.set_page_config(page_title="Nhận diện biển báo giao thông", page_icon="🚦", layout="wide")

# ----------------------------------------------------------------------------
# HÀM TIỆN ÍCH  (nhãn vẽ trên ảnh là mã biển, vd P.102; tên tiếng Việt hiển thị ở bảng)
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Đang tải mô hình YOLO...")
def load_model(weights: str) -> YOLO:
    return YOLO(weights)


def extract_rows(result, frame_idx=None, fps=None):
    """Chuyển kết quả YOLO thành danh sách dict (tên, mã lớp, confidence, bbox)."""
    rows = []
    if result.boxes is None:
        return rows
    for b in result.boxes:
        cid = int(b.cls[0])
        x1, y1, x2, y2 = [int(round(float(v))) for v in b.xyxy[0]]
        row = {
            "Mã biển": result.names[cid],
            "Tên biển báo": VI_NAMES.get(result.names[cid], ""),
            "Mã lớp": cid,
            "Độ tin cậy": round(float(b.conf[0]), 3),
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
        }
        if frame_idx is not None:
            row = {"Khung hình": frame_idx,
                   "Thời điểm (s)": round(frame_idx / fps, 2) if fps else None,
                   "Track ID": None if b.id is None else int(b.id[0]), **row}
        rows.append(row)
    return rows


def sign_label(row) -> str:
    return f'{row["Mã biển"]} – {row["Tên biển báo"]}' if row["Tên biển báo"] else row["Mã biển"]


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8-sig")  # utf-8-sig để Excel đọc đúng tiếng Việt


def show_stats(total: Counter, unique: dict | None = None):
    """Hiển thị thống kê số lượng biển báo."""
    if not total:
        st.info("Không phát hiện được biển báo nào. Hãy thử giảm ngưỡng độ tin cậy.")
        return
    data = {"Biển báo": list(total.keys()), "Số lượt phát hiện": list(total.values())}
    if unique is not None:
        data["Số biển báo (tracking)"] = [len(unique.get(k, ())) for k in total.keys()]
    df = pd.DataFrame(data).sort_values("Số lượt phát hiện", ascending=False).reset_index(drop=True)

    c1, c2 = st.columns(2)
    c1.metric("Tổng số lượt phát hiện", int(sum(total.values())))
    c2.metric("Số loại biển báo", len(total))
    if unique is not None:
        st.caption("«Số biển báo (tracking)» là số biển báo riêng biệt (mỗi biển chỉ đếm 1 lần dù xuất hiện nhiều khung hình).")
    col_a, col_b = st.columns([1, 1])
    col_a.dataframe(df, use_container_width=True, hide_index=True)
    col_b.bar_chart(df.set_index("Biển báo")["Số lượt phát hiện"])


def convert_to_h264(src: str, dst: str) -> str:
    """Mã hóa lại video sang H.264 để trình duyệt phát được. Trả về đường dẫn file dùng được."""
    try:
        import imageio_ffmpeg
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return src
    cmd = [ffmpeg, "-y", "-i", src, "-vcodec", "libx264", "-pix_fmt", "yuv420p",
           "-movflags", "+faststart", "-an", dst]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return dst
    except Exception:
        return src


# ----------------------------------------------------------------------------
# XỬ LÝ ẢNH
# ----------------------------------------------------------------------------
def run_image_mode(model, conf, iou, imgsz):
    files = st.file_uploader("Chọn một hoặc nhiều ảnh", type=["jpg", "jpeg", "png", "bmp", "webp"],
                             accept_multiple_files=True, key="img_up")
    if not files:
        st.info("Hãy tải ảnh lên để bắt đầu nhận diện.")
        return

    total = Counter()
    all_rows = []
    for idx, f in enumerate(files):
        img = np.array(Image.open(f).convert("RGB"))
        bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        t0 = time.time()
        res = model.predict(bgr, conf=conf, iou=iou, imgsz=imgsz, verbose=False)[0]
        dt = time.time() - t0

        annotated = cv2.cvtColor(res.plot(line_width=2), cv2.COLOR_BGR2RGB)
        rows = extract_rows(res)
        for r in rows:
            total[sign_label(r)] += 1
            all_rows.append({"Ảnh": f.name, **r})

        st.subheader(f"🖼️ {f.name}  ·  {len(rows)} biển báo  ·  {dt*1000:.0f} ms")
        c1, c2 = st.columns(2)
        c1.image(img, caption="Ảnh gốc", use_container_width=True)
        c2.image(annotated, caption="Kết quả nhận diện", use_container_width=True)

        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        ok, buf = cv2.imencode(".png", cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
        if ok:
            st.download_button("⬇️ Tải ảnh kết quả", buf.tobytes(),
                               file_name=f"ketqua_{Path(f.name).stem}.png", mime="image/png",
                               key=f"dl_img_{idx}")
        st.divider()

    st.header("📊 Thống kê tổng hợp")
    show_stats(total)
    if all_rows:
        st.download_button("⬇️ Tải bảng kết quả (CSV)", to_csv_bytes(pd.DataFrame(all_rows)),
                           file_name="thong_ke_anh.csv", mime="text/csv", key="dl_csv_img")


# ----------------------------------------------------------------------------
# XỬ LÝ VIDEO
# ----------------------------------------------------------------------------
def process_video(path, weights, conf, iou, imgsz, use_track, preview_every):
    model = YOLO(weights)  # tạo model mới để tracker không bị lẫn giữa các lần chạy
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1

    tmp = tempfile.mkdtemp()
    raw, final = os.path.join(tmp, "raw.mp4"), os.path.join(tmp, "result.mp4")
    writer = cv2.VideoWriter(raw, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    total, unique, rows = Counter(), defaultdict(set), []
    progress, status, preview = st.progress(0.0), st.empty(), st.empty()
    t0, i = time.time(), 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        kw = dict(conf=conf, iou=iou, imgsz=imgsz, verbose=False)
        res = (model.track(frame, persist=True, tracker="bytetrack.yaml", **kw)[0]
               if use_track else model.predict(frame, **kw)[0])

        for r in extract_rows(res, i, fps):
            total[sign_label(r)] += 1
            if r["Track ID"] is not None:
                unique[sign_label(r)].add(r["Track ID"])
            rows.append(r)

        annotated = res.plot(line_width=2)
        writer.write(annotated)
        i += 1
        if i % preview_every == 0:
            preview.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                          caption=f"Khung hình {i}/{n_frames}", use_container_width=True)
        progress.progress(min(i / n_frames, 1.0))
        status.text(f"Đang xử lý: {i}/{n_frames} khung hình · {i/(time.time()-t0+1e-9):.1f} FPS")

    cap.release()
    writer.release()
    status.text("Đang mã hóa video kết quả...")
    out_path = convert_to_h264(raw, final)
    with open(out_path, "rb") as fh:
        video_bytes = fh.read()
    shutil.rmtree(tmp, ignore_errors=True)
    progress.empty(); status.empty(); preview.empty()

    return {"video": video_bytes, "total": total, "unique": dict(unique) if use_track else None,
            "df": pd.DataFrame(rows), "frames": i, "elapsed": time.time() - t0}


def run_video_mode(weights, conf, iou, imgsz, use_track, preview_every):
    f = st.file_uploader("Chọn video", type=["mp4", "avi", "mov", "mkv"], key="vid_up")
    if f is None:
        st.info("Hãy tải video lên để bắt đầu nhận diện.")
        return
    st.video(f)

    if st.button("▶️ Bắt đầu nhận diện video", type="primary"):
        suffix = Path(f.name).suffix or ".mp4"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tf:
            tf.write(f.getvalue())
            src = tf.name
        try:
            st.session_state["video_result"] = process_video(
                src, weights, conf, iou, imgsz, use_track, preview_every)
            st.session_state["video_result"]["name"] = f.name
        finally:
            os.unlink(src)

    res = st.session_state.get("video_result")
    if res:
        st.success(f"Hoàn tất {res['frames']} khung hình trong {res['elapsed']:.1f}s")
        st.subheader("🎬 Video kết quả")
        st.video(res["video"])
        c1, c2 = st.columns(2)
        c1.download_button("⬇️ Tải video kết quả", res["video"],
                           file_name=f"ketqua_{Path(res['name']).stem}.mp4", mime="video/mp4")
        if not res["df"].empty:
            c2.download_button("⬇️ Tải bảng phát hiện (CSV)", to_csv_bytes(res["df"]),
                               file_name="thong_ke_video.csv", mime="text/csv")
        st.header("📊 Thống kê")
        show_stats(res["total"], res["unique"])


# ----------------------------------------------------------------------------
# GIAO DIỆN CHÍNH
# ----------------------------------------------------------------------------
def main():
    st.title("🚦 Hệ thống phát hiện và nhận diện biển báo giao thông bằng YOLO")

    with st.sidebar:
        st.header("⚙️ Cấu hình")
        source = st.radio("Nguồn mô hình", [
            "Mô hình đã huấn luyện (models/best.pt)",
            "Tải lên file trọng số (.pt)",
        ])
        weights = None
        if source.startswith("Mô hình đã"):
            if DEFAULT_WEIGHTS.exists():
                weights = str(DEFAULT_WEIGHTS)
            else:
                st.error("Chưa có models/best.pt. Chạy train.py để huấn luyện, hoặc chọn «Tải lên file trọng số».")
        elif source.startswith("Tải lên"):
            up = st.file_uploader("File .pt", type=["pt"])
            if up:
                p = Path(tempfile.gettempdir()) / f"upload_{up.name}"
                p.write_bytes(up.getvalue())
                weights = str(p)

        conf = st.slider("Ngưỡng độ tin cậy (confidence)", 0.05, 0.95, 0.35, 0.05)
        iou = st.slider("Ngưỡng IoU (NMS)", 0.1, 0.9, 0.45, 0.05)
        imgsz = st.select_slider("Kích thước ảnh đầu vào", [320, 416, 512, 640, 800, 960, 1280], value=960)

        st.subheader("Tùy chọn video")
        use_track = st.checkbox("Theo dõi đối tượng (đếm biển báo duy nhất)", value=True)
        preview_every = st.slider("Xem trước mỗi N khung hình", 1, 30, 5)

    if weights is None:
        st.stop()

    try:
        model = load_model(weights)
    except Exception as e:
        st.error(f"Không tải được mô hình: {e}")
        st.stop()

    with st.sidebar:
        st.caption(f"Số lớp của mô hình: {len(model.names)}")
        with st.expander("Danh sách lớp"):
            st.dataframe(pd.DataFrame(
                [{"Mã lớp": k, "Mã biển": v, "Tên biển báo": VI_NAMES.get(v, "")}
                 for k, v in model.names.items()]), hide_index=True, use_container_width=True)

    tab_img, tab_vid = st.tabs(["🖼️ Nhận diện ảnh", "🎞️ Nhận diện video"])
    with tab_img:
        run_image_mode(model, conf, iou, imgsz)
    with tab_vid:
        run_video_mode(weights, conf, iou, imgsz, use_track, preview_every)


if __name__ == "__main__":
    main()
