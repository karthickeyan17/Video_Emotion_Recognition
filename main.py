import cv2
import numpy as np
import os
import streamlit as st
from PIL import Image
from tensorflow import keras
from tensorflow.keras.models import model_from_json

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
model_path = "cnn_bilstm_model.json"
weights_path = "cnn_bilstm_weights.weights.h5"

# Set page configuration
st.set_page_config(page_title="Emotion Classification in Videos", page_icon="icons8-video-50.png", layout="centered")

try:
    model = keras.models.load_model('cnn_bilstm_model.h5')  
except Exception as e:
    with open(model_path, "r") as json_file:
        model_json = json_file.read()
    model = model_from_json(model_json)
    model.load_weights(weights_path)

emotion_dict = {
    0: "Angry",
    1: "Disgust",
    2: "Fear",
    3: "Happy",
    4: "Sad",
    5: "Surprise",
    6: "Neutral",
}

def preprocess_frame(face):
    gray_face = cv2.resize(face, (48, 48))
    gray_face = gray_face.astype('float32') / 255.0  
    gray_face = np.expand_dims(gray_face, axis=-1)  # Add channel
    return np.expand_dims(gray_face, axis=0)  # Add batch


def classify_emotion(face):
    processed_face = preprocess_frame(face)
    prediction = model.predict(processed_face)
    emotion_index = np.argmax(prediction)
    confidence = np.max(prediction)
    emotion_label = emotion_dict.get(emotion_index)
    return f"{emotion_label} ({confidence * 100:.2f}%)"

st.title("Emotion Classification Through Videos and Images")
if 'webcam_active' not in st.session_state:
    st.session_state.webcam_active = False

option = st.radio("Choose an option:", ("Browse Image", "Real-time Webcam"))

if option == "Browse Image":
    st.write("Upload an image, and the model will detect the face and classify the emotion.")
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = np.array(Image.open(uploaded_file))
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.3, minNeighbors=5)
        for (x, y, w, h) in faces:
            face = gray_image[y:y+h, x:x+w]
            emotion_text = classify_emotion(face)
            cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(image, emotion_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        st.image(image, channels="RGB")

elif option == "Real-time Webcam":
    st.write("Using webcam for real-time emotion classification.")
    if st.button("Start Webcam"):
        st.session_state.webcam_active = True

    if st.button("Stop Webcam"):
        st.session_state.webcam_active = False

    if st.session_state.webcam_active:
        cap = cv2.VideoCapture(0)
        frame_placeholder = st.empty()

        while st.session_state.webcam_active:
            ret, frame = cap.read()
            if not ret:
                break

            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

            for (x, y, w, h) in faces:
                face = gray_frame[y:y+h, x:x+w]
                emotion_text = classify_emotion(face)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                cv2.putText(frame, emotion_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

            frame_placeholder.image(frame, channels="BGR")

        cap.release()
        cv2.destroyAllWindows()
