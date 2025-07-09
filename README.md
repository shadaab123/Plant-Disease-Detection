
# Plant Disease Detection System

A deep learning–based web application for detecting diseases in tomato and corn plants using computer vision.

![Plant Disease Detection](images/image1.png)

---

## Overview

This project provides an AI-powered solution to help farmers and agricultural professionals detect plant diseases early. By leveraging a **ResNet-50 CNN** trained on real-world plant datasets, the system classifies healthy and diseased leaves with high accuracy.

The application is accessible via a **Streamlit web interface**, where users can upload plant leaf images or use their webcam to receive instant diagnoses along with treatment recommendations.

---

## Features

* **Real-Time Diagnosis**: Upload or capture leaf images to get instant results
* **Multi-Crop Support**: Currently supports **tomato** and **corn**
* **Treatment Guidance**: Suggests possible treatments and prevention tips
* **User-Friendly UI**: Built with Streamlit for ease of use
* **Accurate Predictions**: Fine-tuned ResNet-50 model trained on agricultural image datasets

---

## Supported Diseases

### Tomato

* **Early Blight** (*Alternaria solani*)
* **Late Blight** (*Phytophthora infestans*)
* **Healthy**

### Corn

* **Common Rust** (*Puccinia sorghi*)
* **Healthy**

---

## Getting Started

### Prerequisites

* Python 3.8+
* (Optional) CUDA-enabled GPU for faster training

### Installation

1. **Clone the repository**

```bash
git clone https://github.com/your-username/Plant-Disease-Detection.git
cd Plant-Disease-Detection
```

2. **Create a virtual environment**

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Download the trained model**

Place the `plant_disease_model.pth` file in the `models/` directory.
(If training from scratch, see [Training](#training) section below.)

---

## Running the Application

Start the web interface with:

```bash
streamlit run app.py
```

Navigate to `http://localhost:8501` in your browser to interact with the app.

---

## Model Architecture

* **Base Model**: ResNet-50 (ImageNet pre-trained)
* **Custom Classifier**:

  * FC layer (2048 → 512)
  * ReLU activation
  * Dropout (0.3)
  * Output layer (512 → number of classes)
* **Input Size**: 224×224 RGB
* **Augmentations**: Random crop, flip, rotation, color jitter

---

## Training

To train the model from scratch:

```bash
python train.py
```

**Configuration:**

* Optimizer: Adam (lr = 0.001)
* Loss: CrossEntropyLoss
* Batch size: 32
* Epochs: 25 (default, configurable)
* Train/Val Split: 80/20

**Dataset Folder Structure:**

```
data/
├── train/
│   ├── Corn_(maize)___Common_rust_/
│   ├── Corn_(maize)___healthy/
│   ├── Tomato___Early_blight/
│   ├── Tomato___healthy/
│   └── Tomato___Late_blight/
└── test/
    └── (same structure as train)
```

---

## Usage

### Web Interface

1. Launch the app: `streamlit run app.py`
2. Upload an image or use the camera
3. Click **Predict**
4. View disease result and treatment suggestions

### Programmatic Usage

```python
from utils.helpers import load_model, preprocess_image, predict
from PIL import Image

model, class_names = load_model('models/plant_disease_model.pth')
image = Image.open('path/to/image.jpg')
tensor = preprocess_image(image)
label, confidence = predict(tensor, model, class_names)

print(f"Disease: {label}, Confidence: {confidence:.2%}")
```

---

## Project Structure

```
Plant-Disease-Detection/
├── app.py                  # Streamlit application
├── train.py                # Training script
├── model_resnet.ipynb      # Notebook for model experiments
├── requirements.txt        # Project dependencies
├── README.md               # Documentation
├── data/                   # Dataset (train/test folders)
├── models/
│   └── plant_disease_model.pth
├── utils/
│   └── helpers.py          # Preprocessing and prediction utils
├── images/                 # Demo and UI screenshots
└── venv/                   # Virtual environment (optional)
```

---

## Dependencies

Key libraries:

* `torch`, `torchvision` – Model and training
* `streamlit` – Web app
* `opencv-python` – Webcam capture
* `Pillow` – Image handling
* `numpy`, `matplotlib` – Visualization and processing

> See `requirements.txt` for full list.

---

## Demo

![Healthy Tomato](images/tomato_earlybright.png)
*Tomato leaf affected by early blight*

![Diseased Corn](images/rust.png)
*Corn leaf affected by common rust*

![App Interface](images/image2.png)
![App Interface](images/image3.png)

Prediction results in the Streamlit interface

