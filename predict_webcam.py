import cv2
import numpy as np
import tensorflow as tf

# 1. Load Trained Model & Class Labels (alphabetical order from directory)
MODEL_PATH = "models/emotion_model.keras"
EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

model = tf.keras.models.load_model(MODEL_PATH)
face_cascade = cv2.CascadeClassifier(
    "haarcascades/haarcascade_frontalface_default.xml"
)

# 2. Start Webcam Capture
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Could not open webcam.")

print("Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray_frame,
        scaleFactor=1.3,
        minNeighbors=5
    )

    for (x, y, w, h) in faces:
        # Draw bounding box around face
        cv2.rectangle(frame, (x, y - 50), (x + w, y + h + 10), (0, 255, 0), 2)

        # Crop, resize to 48x48, and expand dimensions to (1, 48, 48, 1)
        roi_gray = gray_frame[y:y + h, x:x + w]
        cropped_img = cv2.resize(roi_gray, (48, 48))
        input_tensor = np.expand_dims(np.expand_dims(cropped_img, axis=-1), axis=0)

        # Predict emotion
        predictions = model.predict(input_tensor, verbose=0)
        max_index = int(np.argmax(predictions))
        confidence = predictions[0][max_index] * 100
        label = f"{EMOTIONS[max_index]} ({confidence:.1f}%)"

        cv2.putText(
            frame,
            label,
            (x + 5, y - 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

    cv2.imshow("Facial Emotion Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()