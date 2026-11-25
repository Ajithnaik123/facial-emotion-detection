# Facial Emotion Detection using CNN
**[Live Demo — Try the Facial Emotion Detection App](https://facial-emotion-detection-9giv3vawyqyprtlg6cdne5.streamlit.app/)**
Real-time facial emotion recognition system built with TensorFlow/Keras and OpenCV, trained on the FER-2013 dataset across 7 emotion categories (`Angry`, `Disgust`, `Fear`, `Happy`, `Neutral`, `Sad`, `Surprise`).

## Project Structure
- `train.py` — Builds and trains a 4-block Convolutional Neural Network on 48×48 grayscale facial images.
- `predict_webcam.py` — Runs real-time face detection (Haar Cascade) and emotion classification via webcam.
- `models/` — Stores the saved Keras model (`emotion_model.keras`).
- `haarcascades/` — Contains OpenCV's frontal face detector XML.

## Setup & Usage
1. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
