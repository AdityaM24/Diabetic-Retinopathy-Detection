# 🩺 Diabetic Retinopathy Detection using Transfer Learning

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)

A deep learning project that detects Diabetic Retinopathy (DR) from retinal fundus images using MobileNetV2 with transfer learning, featuring data augmentation, class balancing, and explainability tools like Grad-CAM.  

</div>

---

## 📋 Table of Contents

- [Overview](#overview)
- [What is Diabetic Retinopathy?](#what-is-diabetic-retinopathy)
- [Features](#features)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Model Explainability](#model-explainability)
- [Technologies Used](#technologies-used)
- [Contributing](#contributing)
- [Acknowledgments](#acknowledgments)

---

## 🎯 Overview

This project implements an end-to-end computer vision pipeline for **binary classification** of retinal fundus images to detect Diabetic Retinopathy.  The system leverages **MobileNetV2**, a lightweight and efficient convolutional neural network pre-trained on ImageNet, making it suitable for deployment in resource-constrained environments.

### Key Highlights:
- ✅ Binary classification:  **DR** vs **No DR**
- ✅ Transfer learning with MobileNetV2
- ✅ Data preprocessing with Gaussian filtering
- ✅ Advanced data augmentation
- ✅ Class balancing with computed class weights
- ✅ Model explainability using Grad-CAM
- ✅ TensorBoard integration for training monitoring

---

## 🔬 What is Diabetic Retinopathy?

**Diabetic Retinopathy (DR)** is a diabetes complication that affects the eyes. It's caused by damage to the blood vessels of the light-sensitive tissue at the back of the eye (retina). Early detection is crucial as it can lead to blindness if left untreated. 

### Why Automated Detection? 
- 📊 **Early Detection**: Automated screening can help identify DR in early stages
- ⚡ **Speed**: Process thousands of images quickly
- 🎯 **Accuracy**: Deep learning models can achieve diagnostic-level performance
- 💰 **Cost-Effective**:  Reduces the need for extensive manual screening

---

## ✨ Features

- **Transfer Learning**: Utilizes pre-trained MobileNetV2 for efficient feature extraction
- **Data Augmentation**: Implements real-time augmentation (rotation, shift, zoom, brightness) to improve generalization
- **Gaussian Filtering**: Preprocessing pipeline with Gaussian blur for noise reduction
- **Class Balancing**: Handles imbalanced datasets using computed class weights
- **Model Evaluation**: Comprehensive metrics including accuracy, precision, recall, and F1-score
- **Grad-CAM Visualization**: Explains model predictions with heatmap overlays
- **TensorBoard Integration**: Real-time training monitoring and visualization

---

## 🏗️ Model Architecture

### Base Model: MobileNetV2
- **Pre-trained on**:  ImageNet
- **Input Shape**: 224×224×3
- **Frozen Layers**: All base layers (feature extraction only)

### Custom Classification Head
```
MobileNetV2 (frozen)
    ↓
GlobalAveragePooling2D
    ↓
BatchNormalization
    ↓
Dropout (0.3)
    ↓
Dense (128 units, ReLU)
    ↓
Dropout (0.3)
    ↓
Dense (2 units, softmax)
```

### Training Configuration
- **Image Size**: 224×224 pixels
- **Batch Size**: 32
- **Epochs**: 20
- **Optimizer**: Adam (learning rate = 1e-4)
- **Loss Function**: Categorical Crossentropy
- **Class Weights**: Computed using sklearn's `compute_class_weight('balanced')`
  - DR class: ~1.42
  - No_DR class:  ~0.77

### Data Augmentation Parameters
- Rotation range: ±25°
- Width/Height shift: 0.1
- Shear range: 0.05
- Zoom range: 0.1
- Horizontal flip: True
- Brightness range: (0.8, 1.2)
- Fill mode: Nearest

---

## 📊 Dataset

**Source**: [Kaggle - Diabetic Retinopathy 224x224 Gaussian Filtered](https://www.kaggle.com/datasets/sovitrath/diabetic-retinopathy-224x224-gaussian-filtered)

The dataset consists of Gaussian-filtered retinal fundus images organized into three splits: 

### Dataset Statistics
| Split | DR Images | No_DR Images | Total |
|-------|-----------|--------------|-------|
| Train | 684 | 1,263 | 1,947 |
| Validation | 279 | 271 | 550 |
| Test | 279 | 271 | 550 |

### Directory Structure
```
dataset/
├── train/
│   ├── DR/          # Images with Diabetic Retinopathy
│   └── No_DR/       # Healthy retinal images
├── val/
│   ├── DR/
│   └── No_DR/
└── test/
    ├── DR/
    └── No_DR/
```

### Preprocessing Steps
1. **Gaussian Filtering**: Pre-applied to reduce noise
2. **Resizing**: All images are 224×224 pixels
3. **Normalization**: Pixel values rescaled to [0, 1] range
4. **Augmentation**: Applied only during training

---

## 📁 Project Structure

```
Diabetic-Retinopathy-Detection/
├── README.md                      # Project documentation
├── improv.ipynb                   # Main Jupyter notebook with full pipeline
├── mobilenetv2_dr_imp.h5         # Trained model weights
├── . gitignore                     # Git ignore file
├── gaussian_filtered_images/      # Preprocessed images directory
├── logs/                          # TensorBoard logs
│   └── mobilenetv2/
├── train/                         # Training dataset
│   ├── DR/                        # 684 images
│   └── No_DR/                     # 1,263 images
├── val/                           # Validation dataset
│   ├── DR/                        # 279 images
│   └── No_DR/                     # 271 images
└── test/                          # Test dataset
    ├── DR/                        # 279 images
    └── No_DR/                     # 271 images
```

---

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- (Optional) CUDA-enabled GPU for faster training

### Setup Instructions

1. **Clone the repository**
```bash
git clone https://github.com/AdityaM24/Diabetic-Retinopathy-Detection.git
cd Diabetic-Retinopathy-Detection
```

2. **Create a virtual environment** (recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install required packages**
```bash
pip install tensorflow numpy pandas matplotlib seaborn scikit-learn opencv-python jupyter
```

### Required Libraries
```python
tensorflow>=2.0
numpy
pandas
matplotlib
seaborn
scikit-learn
opencv-python (cv2)
jupyter
```

---

## 💻 Usage

### 1. Running the Notebook

Open and run the Jupyter notebook:
```bash
jupyter notebook improv.ipynb
```

The notebook includes:
- ✅ Dataset loading and information display
- ✅ Data cleaning (checking for corrupted images)
- ✅ Exploratory Data Analysis (EDA)
- ✅ Class distribution visualization
- ✅ Sample image visualization
- ✅ Data augmentation setup
- ✅ Model building and compilation
- ✅ Training with callbacks (ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard)
- ✅ Evaluation and metrics
- ✅ Confusion matrix and classification report
- ✅ Grad-CAM visualization for explainability

### 2. Making Predictions

```python
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the trained model
model = load_model('mobilenetv2_dr_imp.h5')

# Load and preprocess an image
img_path = 'path/to/retinal/image.jpg'
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Make prediction
prediction = model.predict(img_array)
class_names = ['DR', 'No_DR']
predicted_class = class_names[np.argmax(prediction)]
confidence = np.max(prediction) * 100

print(f"Prediction: {predicted_class}")
print(f"Confidence:  {confidence:.2f}%")
```

### 3. Monitoring Training with TensorBoard

```bash
tensorboard --logdir=logs/mobilenetv2
```

Then open your browser and navigate to `http://localhost:6006`

---

## 🔍 Model Explainability

The project implements **Grad-CAM (Gradient-weighted Class Activation Mapping)** to visualize which regions of the retinal image the model focuses on when making predictions. 

### Why Grad-CAM?
- 🔎 **Transparency**:  Understand what features the model uses for classification
- 🏥 **Clinical Trust**: Build confidence with medical professionals
- 🐛 **Debugging**: Identify if the model is learning correct features
- ⚖️ **Bias Detection**: Spot potential issues in model decision-making

---

## 🛠️ Technologies Used

- **Framework**: TensorFlow / Keras
- **Base Model**: MobileNetV2 (ImageNet pre-trained)
- **Language**: Python 3.8+
- **Data Processing**: NumPy, Pandas, OpenCV
- **Visualization**:  Matplotlib, Seaborn
- **Evaluation**: Scikit-learn
- **Logging**: TensorBoard
- **Development**: Jupyter Notebook

---

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. Fork the repository
2. Create a new branch (`git checkout -b feature/improvement`)
3. Make your changes
4. Commit your changes (`git commit -am 'Add new feature'`)
5. Push to the branch (`git push origin feature/improvement`)
6. Create a Pull Request

### Ideas for Improvement:
- [ ] Multi-class classification (different DR severity levels:  No DR, Mild, Moderate, Severe, Proliferative)
- [ ] Model optimization for mobile deployment (TensorFlow Lite)
- [ ] Web application for easy inference (Flask/FastAPI + Streamlit)
- [ ] Support for other architectures (EfficientNet, ResNet, Vision Transformers)
- [ ] Advanced augmentation techniques (Mixup, CutMix)
- [ ] Ensemble methods for improved accuracy
- [ ] Cross-validation implementation
- [ ] SHAP values for additional explainability

---

## 🙏 Acknowledgments

- **Dataset**: Kaggle community for providing the Gaussian-filtered diabetic retinopathy dataset
- **TensorFlow/Keras**: For the excellent deep learning framework
- **MobileNetV2**:  Efficient architecture developers (Google Research)
- **Medical Imaging Community**: For advancing AI in healthcare

---

## 📧 Contact

**Aditya Mahale** - [@AdityaM24](https://github.com/AdityaM24)

Project Link: [https://github.com/AdityaM24/Diabetic-Retinopathy-Detection](https://github.com/AdityaM24/Diabetic-Retinopathy-Detection)

---

<div align="center">

⭐ **If you find this project helpful, please consider giving it a star! ** ⭐

</div>
