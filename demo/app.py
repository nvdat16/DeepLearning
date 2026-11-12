"""Web demo object tracking trên VisDrone, gọi backend FastAPI chạy trên Colab qua ngrok.

    pip install -r demo/requirements.txt
    streamlit run demo/app.py

Chạy demo/Tracking_Server.ipynb trên Colab trước, rồi dán Public URL (ngrok) vào sidebar.
Có thể đặt sẵn URL bằng biến môi trường TRACKING_API_URL.
"""
import os

import pandas as pd
import requests
import streamlit as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VIDEO_DIR = os.path.join(ROOT, "video")
# bỏ qua trang cảnh báo của ngrok bản free
HEADERS = {"ngrok-skip-browser-warning": "true"}
TRACKERS = {"BoT-SORT": "botsort", "ByteTrack": "bytetrack"}

st.set_page_config(page_title="VisDrone Tracking Demo", layout="wide")
st.title("Object Tracking trên VisDrone")

with st.sidebar:
    st.header("Cấu hình")
    api_url = st.text_input("API URL (ngrok)", os.environ.get("TRACKING_API_URL", ""),
                            placeholder="https://xxxx.ngrok-free.app").strip().rstrip("/")
    if api_url and not api_url.startswith(("http://", "https://")):
        st.error("API URL phải có dạng https://xxxx.ngrok-free.app (lấy ở output cell ngrok trên Colab), "
                 "không phải ngrok authtoken")
        api_url = ""
    if st.button("Kiểm tra kết nối", disabled=not api_url):
        try:
            r = requests.get(f"{api_url}/", headers=HEADERS, timeout=15)
            r.raise_for_status()
            info = r.json()
            st.success(f"Kết nối OK, model: {info.get('model')}, device: {info.get('device')}")
        except Exception as e:
            st.error(f"Không kết nối được: {e}")

    tracker_label = st.selectbox("Tracker", list(TRACKERS))
    conf = st.slider("Confidence", 0.05, 0.9, 0.1, 0.05,
                     help="Để thấp (~0.1): tracker tự lọc 2 tầng, giữ được object nhỏ")
    imgsz = st.select_slider("Image size", [640, 960, 1280], 640,
                             help="Nên bằng kích thước lúc train")

source = st.radio("Nguồn video", ["Upload", "Video mẫu"], horizontal=True)
video_name, video_bytes = None, None
if source == "Upload":
    uploaded = st.file_uploader("Chọn video", type=["mp4", "avi", "mov"])
    if uploaded:
        video_name, video_bytes = uploaded.name, uploaded.getvalue()
else:
    samples = sorted(f for f in os.listdir(VIDEO_DIR) if f.endswith(".mp4")) if os.path.isdir(VIDEO_DIR) else []
    if samples:
        video_name = st.selectbox("Video", samples)
        with open(os.path.join(VIDEO_DIR, video_name), "rb") as f:
            video_bytes = f.read()
    else:
        st.info(f"Không có video mẫu trong {VIDEO_DIR}")

if video_bytes and len(video_bytes) > 100 * 1024 * 1024:
    st.warning(f"Video nặng {len(video_bytes) / 1024 ** 2:.0f} MB: upload qua ngrok và tracking sẽ rất lâu. "
               "Nên thử video nhỏ trước (vd uav0000305_00000_v.mp4).")

if st.button("Chạy tracking", type="primary", disabled=not (api_url and video_bytes)):
    with st.spinner("Đang tracking trên Colab..."):
        try:
            r = requests.post(
                f"{api_url}/track",
                headers=HEADERS,
                files={"file": (video_name, video_bytes, "video/mp4")},
                data={"tracker": TRACKERS[tracker_label], "conf": conf, "imgsz": imgsz},
                timeout=1800,
            )
            r.raise_for_status()
            res = r.json()
            if "error" in res:
                st.error(res["error"])
            else:
                v = requests.get(f"{api_url}/video/{res['job_id']}", headers=HEADERS, timeout=600)
                v.raise_for_status()
                # video gốc mp4v không phát được trên trình duyệt -> lấy bản H.264 server đã encode
                orig = requests.get(f"{api_url}/video/{res['job_id']}", params={"original": "true"},
                                    headers=HEADERS, timeout=600)
                ok = orig.ok and orig.headers.get("content-type", "").startswith("video/")
                # lưu vào session để không mất kết quả khi Streamlit rerun (vd bấm tải về)
                st.session_state.result = res
                st.session_state.result_video = v.content
                st.session_state.input_video = orig.content if ok else video_bytes
                st.session_state.input_name = video_name
        except requests.RequestException as e:
            st.error(f"Lỗi gọi API: {e}")

res = st.session_state.get("result")
if res:
    left, right = st.columns(2)
    with left:
        st.subheader("Video gốc")
        st.video(st.session_state.input_video)
    with right:
        st.subheader(f"Kết quả: {res['model']} + {res['tracker']}")
        st.video(st.session_state.result_video)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Unique IDs", res["unique_ids"])
    m2.metric("Frames", res["frames"])
    m3.metric("FPS xử lý", res["fps"])
    m4.metric("Thời gian (s)", res["time_s"])

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Số ID duy nhất theo class**")
        st.bar_chart(pd.Series(res["per_class"], name="IDs"))
    with c2:
        st.markdown("**Số object theo frame**")
        st.line_chart(pd.DataFrame({"objects": res["objects_per_frame"]}))

    stem = os.path.splitext(st.session_state.input_name)[0]
    st.download_button("Tải video kết quả", st.session_state.result_video,
                       file_name=f"{stem}_{res['model']}_{res['tracker']}.mp4", mime="video/mp4")
