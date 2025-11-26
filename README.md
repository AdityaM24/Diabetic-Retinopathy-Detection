# Diabetic Retinopathy Detection using Transfer Learning (MobileNetV2)
A deep learning project that detects Diabetic Retinopathy (DR) from retinal fundus images using MobileNetV2, trained with data augmentation, class balancing, and explainability tools like Grad-CAM.

## 🧠Brief Overview

This project builds an end-to-end computer vision pipeline to classify retinal images into:
- DR (Diabetic Retinopathy)
  
- No DR

The model uses MobileNetV2 with custom classification layers, trained on a curated retina dataset. The system includes preprocessing, EDA, augmentation, model training, evaluation, and explainability.

## ✏️Model Architecture

Base Model: MobileNetV2 (ImageNet pretrained, frozen)

Custom Layers:
- Global Average Pooling

- Dropout (0.3)

- Dense Softmax (2 classes)

Optimizer: Adam (lr=1e-4)

Loss: Categorical Crossentropy
