import streamlit as st
import torch
import cv2
import numpy as np
from torchvision import transforms
from nn_model import EnhancedCNNMoreDropout

WEIGHTS_PATH = 'weights.pth'

def preprocess(image):
    preprocessed = cv2.resize(image, (128, 128))
    preprocessed = cv2.cvtColor(preprocessed, cv2.COLOR_BGR2GRAY)
    _, preprocessed = cv2.threshold(preprocessed, 127, 1, cv2.THRESH_BINARY)
    preprocessed = preprocessed.astype(np.float32)
    return preprocessed

def image_to_tensor(image):
    tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0)
    return tensor

@st.cache_data
def load_model(path):
    weights = torch.load(path, map_location=torch.device('cpu'))
    model = EnhancedCNNMoreDropout()
    model.load_state_dict(weights)
    model.eval()
    return model

def predict(image, model):
    tensor = image_to_tensor(image)
    model.eval()
    with torch.inference_mode():
        result = model(tensor)
    return result

st.title("🐱 or 🐶 | Sketch Classification")
input_file = st.file_uploader("Upload an image", type=["jpg", "png"])

if input_file is not None:
    file_bytes = np.asarray(bytearray(input_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    st.image(img, caption='Uploaded Image', channels="BGR", use_column_width=True)
    
    preprocessed = preprocess(img)
    model = load_model(WEIGHTS_PATH)
    result = predict(image_to_tensor(preprocessed), model)
    result = result.squeeze().item()
    emoji = "🐱" if result >= 0.5 else "🐶"
    
    st.header(f"I believe this sketch contains a {emoji}")