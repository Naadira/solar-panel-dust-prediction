import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import h5py as h5
from tensorflow.keras.models import load_model
import subprocess
import os

# Check if model exists, download it if not
if not os.path.isfile('model.h5'):
    subprocess.run(['curl --output model.h5 "https://media.githubusercontent.com/media/MuhammadAqhariNasrin/Optimizing-Solar-Panel-Efficiency-Computer-Vision-Approach-to-Dust-Detection-on-Solar-Panel/main/solarpanelimageclassifier.h5"'], shell=True)

model = tf.keras.models.load_model('solarpanelimageclassifier.h5', compile=False)

# Streamlit app
st.set_page_config(page_title="Solar Panel Classifier", page_icon="🌞", layout="wide")
st.title("Optimizing Solar Panel Efficiency: Computer Vision Approach to Dust Detection")
st.markdown("""
    This app uses a **Deep Learning model** to detect whether a solar panel is **clean** or **dusty** based on the uploaded image.
    ### Instructions:
    1. Upload an image of a solar panel.
    2. Wait for the model to process and classify the panel as **Clean** or **Dusty**.
    3. You will get the prediction result along with the classification.

    The model analyzes the image and provides predictions to help optimize solar panel efficiency.
""")

# Upload image through Streamlit interface
uploaded_file = st.file_uploader("Choose an image of a solar panel...", type=["jpg", "jpeg", "png"])

# Function to make predictions
def make_prediction(image):
    resize = tf.image.resize(image, (256, 256))
    yhat = model.predict(np.expand_dims(resize / 255, 0))
    return yhat[0][0]

# Display results
if uploaded_file is not None:
    st.image(uploaded_file, caption='Uploaded Image.', use_container_width=True)
    st.write("### Classifying...")

    # Read the image using PIL
    img = Image.open(uploaded_file)

    # Convert the image to a numpy array
    img_array = np.array(img)

    # Make prediction
    prediction = make_prediction(img_array)

    # Display the results
    if prediction > 0.5:
        st.write("### Prediction: Dusty Solar Panel")
        st.warning("The solar panel is dusty. Consider cleaning it for optimal performance!")
    else:
        st.write("### Prediction: Clean Solar Panel")
        st.success("The solar panel is clean and operating at its best efficiency!")

# Add some additional styling and information
st.markdown("""
    ---
    ## About the Project:
    This project leverages **Computer Vision** and **Deep Learning** to automatically assess the cleanliness of solar panels.
    - **Why it matters:** Dust accumulation on solar panels reduces their efficiency. Regular cleaning ensures optimal performance.
    - The model has been trained to differentiate between clean and dusty solar panels using high-quality image data.
    
    ### Download Sample Data
    You can explore some sample data and the full project [here](https://drive.google.com/drive/folders/12Q3MBI8SPw0vHsO_kkS5izkxw0F7tXx4).
""")
