# Emotion Classification in Videos and Images

## Overview
This project is a deep learning-based application for emotion recognition in videos and images. It uses a Convolutional Neural Network (CNN) combined with a Bidirectional Long Short-Term Memory (BiLSTM) network to classify emotions from facial expressions. The model is integrated into a Streamlit-based web application for ease of use.

## Model Development

### Dataset
The model was trained on the **FER-2013** dataset, which contains grayscale images of faces labeled with seven emotions:
- Angry
- Disgust
- Fear
- Happy
- Sad
- Surprise
- Neutral

### Preprocessing
1. **Face Detection:**
   - Used OpenCV's Haar Cascade Classifier (`haarcascade_frontalface_default.xml`) to detect faces in images.
   
2. **Image Processing:**
   - Converted images to grayscale.
   - Resized images to `48x48` pixels.
   - Normalized pixel values between `0` and `1`.
   
### Model Architecture
The model consists of:
1. **CNN Layers:**
   - Extracts spatial features from facial expressions.
   - Includes convolutional layers followed by max pooling.
   
2. **BiLSTM Layers:**
   - Captures temporal dependencies in facial expressions.
   - Helps in better classification by learning sequential patterns.
   
3. **Fully Connected Layers:**
   - Maps extracted features to emotion categories.
   - Uses softmax activation for classification.

### Training
- Optimizer: **Adam**
- Loss Function: **Categorical Crossentropy**
- Batch Size: **32**
- Epochs: **50**
- Validation Split: **20%**

### Implementation
The trained model is stored in two files:
- `cnn_bilstm_model.json`: Stores the model architecture.
- `cnn_bilstm_weights.h5`: Stores the trained weights.

In the application, the model is loaded using TensorFlow and Keras:
```python
from tensorflow.keras.models import model_from_json

# Load model architecture
with open("cnn_bilstm_model.json", "r") as json_file:
    model_json = json_file.read()
model = model_from_json(model_json)

# Load model weights
model.load_weights("cnn_bilstm_weights.h5")
```

## Features
- **Image Upload:**
  - Users can upload an image to detect emotions in detected faces.
- **Real-time Webcam Detection:**
  - The application can analyze emotions from a live webcam feed.
- **User-friendly UI:**
  - Built using Streamlit for a seamless experience.

## Demo
Here is an image of friends testing the Streamlit application:
![Demo](https://github.com/karthickeyan17/Video_Emotion_Recognition/blob/main/Picture1.jpg)

Let me know if you need any further modifications! 🚀

