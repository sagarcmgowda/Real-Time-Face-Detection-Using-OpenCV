import streamlit as st
import cv2 as cv
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration

face_detect = cv.CascadeClassifier(
    cv.data.haarcascades + "haarcascade_frontalface_default.xml"
)

st.set_page_config(page_title="Real-Time Face Detection")
st.title("🎥 Real-Time Face Detection (Webcam)")

RTC_CONFIGURATION = RTCConfiguration(
    {
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]},
            {"urls": ["stun:stun1.l.google.com:19302"]},
        ]
    }
)

class FaceDetector(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        faces = face_detect.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cv.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

        return img

webrtc_streamer(
    key="face-detection",
    video_transformer_factory=FaceDetector,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": True, "audio": False},
)
