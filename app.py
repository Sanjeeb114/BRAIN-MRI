import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# Load the trained model
model = load_model("/content/best_model.h5")

# Define class names (modify based on your model)
class_names = ["glioma", "meningioma", "notumor", "pituitary"]

st.title("MRI Image Analysis")

uploaded_file = st.file_uploader("Upload an MRI Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)  # ✅ Fixed

    # Preprocess the image
    image = image.resize((168, 168))  # Adjust size based on your model input
    image = np.array(image) / 255.0  # Normalize
    image = np.expand_dims(image, axis=0)  # Add batch dimension

    # Make prediction
    prediction = model.predict(image)
    predicted_class = np.argmax(prediction)

    st.write(f"Prediction: **{class_names[predicted_class]}**")
