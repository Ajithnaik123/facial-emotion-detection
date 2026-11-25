import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# 1. Load Trained Emotion Model & Class Labels
MODEL_PATH = "models/emotion_model.keras"
EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

model = tf.keras.models.load_model(MODEL_PATH)

# 2. Initialize MediaPipe Tasks FaceDetector with higher confidence threshold
base_options = python.BaseOptions(model_asset_path="haarcascades/blaze_face_short_range.tflite")
options = vision.FaceDetectorOptions(
    base_options=base_options,
    min_detection_confidence=0.75  # Raised from 0.5 to ignore background objects
)
detector = vision.FaceDetector.create_from_options(options)

# 3. Start Webcam Capture
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Could not open webcam.")

print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h_frame, w_frame = frame.shape[:2]
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    detection_result = detector.detect(mp_image)

    for detection in detection_result.detections:
        # Ensure detection confidence is strong and facial keypoints are present
        score = detection.categories[0].score if detection.categories else 0.0
        if score < 0.75 or not detection.keypoints:
            continue

        bbox = detection.bounding_box
        x1 = max(0, bbox.origin_x)
        y1 = max(0, bbox.origin_y)
        x2 = min(w_frame, bbox.origin_x + bbox.width)
        y2 = min(h_frame, bbox.origin_y + bbox.height)

        box_w = x2 - x1
        box_h = y2 - y1
        if box_w < 60 or box_h < 60:
            continue

        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Crop face, resize to 48x48 grayscale, and predict emotion
        roi_gray = gray_frame[y1:y2, x1:x2]
        cropped_img = cv2.resize(roi_gray, (48, 48))
        input_tensor = np.expand_dims(np.expand_dims(cropped_img, axis=-1), axis=0)

        predictions = model.predict(input_tensor, verbose=0)
        max_index = int(np.argmax(predictions))
        emotion_conf = predictions[0][max_index] * 100
        label = f"{EMOTIONS[max_index]} ({emotion_conf:.1f}%)"

        label_y = y1 - 10 if y1 - 10 > 20 else y1 + 25
        cv2.putText(
            frame,
            label,
            (x1, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )

    cv2.imshow("Facial Emotion Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

detector.close()
cap.release()
cv2.destroyAllWindows()