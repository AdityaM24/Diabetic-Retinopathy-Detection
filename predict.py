import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import argparse

def predict_image(image_path, model_path='mobilenetv2_dr_imp.h5'):
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found.")
        return

    if not os.path.exists(image_path):
        print(f"Error: Image file '{image_path}' not found.")
        return

    try:
        # Load the trained model
        print(f"Loading model from {model_path}...")
        model = load_model(model_path)

        # Load and preprocess an image
        print(f"Processing image {image_path}...")
        img = image.load_img(image_path, target_size=(224, 224))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Make prediction
        prediction = model.predict(img_array)
        class_names = ['DR', 'No_DR']
        
        # Get the predicted class index
        predicted_idx = np.argmax(prediction)
        predicted_class = class_names[predicted_idx]
        confidence = np.max(prediction) * 100

        print("-" * 30)
        print(f"Prediction: {predicted_class}")
        print(f"Confidence: {confidence:.2f}%")
        print("-" * 30)
        print(f"Raw probabilities: {prediction}")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict Diabetic Retinopathy from an image.')
    parser.add_argument('--image', type=str, default='test/DR/00a8624548a9.png', help='Path to the image file')
    parser.add_argument('--model', type=str, default='mobilenetv2_dr_imp.h5', help='Path to the model file')
    
    args = parser.parse_args()
    
    predict_image(args.image, args.model)
