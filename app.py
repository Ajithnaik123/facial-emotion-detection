import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from PIL import Image

st.set_page_config(page_title="Facial Emotion Detection", page_icon="😊", layout="centered")

MODEL_PATH = "models/emotion_model.keras"
DETECTOR_PATH = "haarcascades/blaze_face_short_range.tflite"
EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]


@st.cache_resource
def load_models():
    emotion_model = tf.keras.models.load_model(MODEL_PATH)
    base_options = python.BaseOptions(model_asset_path=DETECTOR_PATH)
    options = vision.FaceDetectorOptions(base_options=base_options, min_detection_confidence=0.75)
    face_detector = vision.FaceDetector.create_from_options(options)
    return emotion_model, face_detector


def analyze_image(image_pil, emotion_model, face_detector):
    rgb_frame = np.array(image_pil.convert("RGB"))
    annotated = rgb_frame.copy()
    gray_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2GRAY)
    h_frame, w_frame = rgb_frame.shape[:2]

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    detection_result = face_detector.detect(mp_image)

    results_list = []
    for detection in detection_result.detections:
        bbox = detection.bounding_box
        x1 = max(0, bbox.origin_x)
        y1 = max(0, bbox.origin_y)
        x2 = min(w_frame, bbox.origin_x + bbox.width)
        y2 = min(h_frame, bbox.origin_y + bbox.height)

        if x2 <= x1 or y2 <= y1:
            continue

        roi_gray = gray_frame[y1:y2, x1:x2]
        cropped_img = cv2.resize(roi_gray, (48, 48))
        input_tensor = np.expand_dims(np.expand_dims(cropped_img, axis=-1), axis=0)

        predictions = emotion_model.predict(input_tensor, verbose=0)[0]
        max_index = int(np.argmax(predictions))
        confidence = float(predictions[max_index] * 100)
        emotion = EMOTIONS[max_index]

        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{emotion} ({confidence:.1f}%)"
        label_y = y1 - 10 if y1 - 10 > 20 else y1 + 25
        cv2.putText(
            annotated,
            label,
            (x1, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )

        probs = {EMOTIONS[i]: float(predictions[i] * 100) for i in range(len(EMOTIONS))}
        results_list.append((emotion, confidence, probs))

    return annotated, results_list


# --- UI Layout ---
st.title("Facial Emotion Detection")
st.write("Detect facial expressions across 7 emotions (`Angry`, `Disgust`, `Fear`, `Happy`, `Neutral`, `Sad`, `Surprise`) using a custom CNN trained on FER-2013 and MediaPipe BlazeFace.")

emotion_model, face_detector = load_models()

mode = st.radio("Choose Input Mode:", ["Webcam Capture", "Upload an Image"], horizontal=True)

input_image = None
if mode == "Webcam Capture":
    camera_photo = st.camera_input("Take a photo using your webcam")
    if camera_photo is not None:
        input_image = Image.open(camera_photo)
else:
    uploaded_file = st.file_uploader("Upload a face image (JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        input_image = Image.open(uploaded_file)

if input_image is not None:
    annotated_img, detections = analyze_image(input_image, emotion_model, face_detector)
    st.image(annotated_img, caption="Detected Emotion", use_container_width=True)

    if not detections:
        st.warning("No face detected. Try better lighting or face the camera directly.")
    else:
        for idx, (emotion, confidence, probs) in enumerate(detections, start=1):
            st.success(f"**Face {idx}:** {emotion} ({confidence:.1f}% confidence)")
            st.bar_chart(probs)