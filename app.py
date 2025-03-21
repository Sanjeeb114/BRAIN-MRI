import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# Load the trained model
model = load_model("best_model.h5")  # Ensure this file is uploaded

# Class labels (Update these based on your classes)
class_labels = ["glioma", "meningioma", "notumor", "pituitary"]  # Modify as per your model

# Streamlit UI
st.title("MRI Image Classification")
st.write("Upload an MRI image to classify")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Preprocess the image
    img = image.resize((224, 224))  # Adjust based on your model input size
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Make prediction
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)[0]

    # Show result
    st.write(f"Prediction: **{class_labels[predicted_class]}**")
