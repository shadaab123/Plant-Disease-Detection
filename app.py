import streamlit as st
from PIL import Image
import numpy as np
import torch
import cv2
from utils.helpers import load_model, preprocess_image, predict

# Load model
model, class_names = load_model('models/plant_disease_model.pth')

# Disease information
disease_info = {
    'Tomato___Late_blight': {
        'treatment': 'Apply fungicides containing chlorothalonil or mancozeb',
        'prevention': 'Ensure proper spacing and avoid overhead watering'
    },
    'Tomato___Early_blight': {
        'treatment': 'Use copper-based fungicides or chlorothalonil',
        'prevention': 'Rotate crops and remove infected plant debris'
    },
    'Tomato___Bacterial_spot': {
        'treatment': 'Apply copper sprays mixed with mancozeb',
        'prevention': 'Use disease-free seeds and practice crop rotation'
    },
    'Corn___Common_rust': {
        'treatment': 'Apply fungicides like propiconazole',
        'prevention': 'Plant resistant varieties and ensure good air circulation'
    },
    # Add healthy cases
    'Tomato___Healthy': {
        'message': 'Your tomato plant looks healthy! Maintain good practices.'
    },
    'Corn___Healthy': {
        'message': 'Your corn plant looks healthy! Keep up the good work.'
    }
}

# Streamlit UI
st.title('Plant Disease Detection')
st.write('Upload an image of tomato or corn plant for disease diagnosis')

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
camera_image = st.camera_input("Or take a picture")

image_source = uploaded_file or camera_image

if image_source:
    # Convert to numpy array
    image = Image.open(image_source)
    image = np.array(image)
    
    # Display image
    st.image(image, caption='Uploaded Image', width=300)
    
    # Preprocess and predict
    if st.button('Predict'):
        with st.spinner('Analyzing...'):
            # Convert to RGB if needed
            if len(image.shape) == 2:  # Grayscale
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif image.shape[2] == 4:  # RGBA
                image = image[:, :, :3]
            
            # Preprocess and predict
            tensor_image = preprocess_image(image)
            prediction, confidence = predict(tensor_image, model, class_names)
            
            # Display results
            st.success(f"Prediction: {prediction.replace('___', ' ')} (Confidence: {confidence:.2%})")
            
            # Show treatment/prevention info
            if prediction in disease_info:
                info = disease_info[prediction]
                if 'treatment' in info:
                    st.subheader("Recommended Treatment")
                    st.write(info['treatment'])
                    st.subheader("Prevention Tips")
                    st.write(info['prevention'])
                else:
                    st.write(info['message'])