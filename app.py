import streamlit as st
import cv2 as cv
import av
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# Load Haar Cascade
face_detect = cv.CascadeClassifier(
    cv.data.haarcascades + "haarcascade_frontalface_default.xml"
)

st.set_page_config(page_title="Face Detection", layout="centered")
st.title("🎥 Real-Time Face Detection (Webcam)")

class FaceDetector(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        faces = face_detect.detectMultiScale(
            gray,
            scaleFactor=1.3,
            minNeighbors=5
        )

        for (x, y, w, h) in faces:
            cv.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

        return img

webrtc_streamer(
    key="face-detection",
    video_transformer_factory=FaceDetector,
    media_stream_constraints={
        "video": True,
        "audio": False
    },
)
