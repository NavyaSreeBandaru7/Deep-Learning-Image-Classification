import tensorflow as tf
import numpy as np
from PIL import Image
import sys
import os

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck']

def load_model(model_path='../saved_models/cifar10_cnn_model.h5'):
    """Load trained model"""
    return tf.keras.models.load_model(model_path)

def preprocess_image(image_path):
    """Load and preprocess image for prediction"""
    img = Image.open(image_path)
    img = img.resize((32, 32))
    
    # Convert to RGB if not
    if img.mode != 'RGB':
        img = img.convert('RGB')
        
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

def predict_image(model, image_path):
    """Make prediction on single image"""
    preprocessed_img = preprocess_image(image_path)
    prediction = model.predict(preprocessed_img)
    predicted_class = np.argmax(prediction)
    confidence = np.max(prediction) * 100
    
    return class_names[predicted_class], confidence

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python predict.py <image_path> [model_path]")
        sys.exit(1)
    
    image_path = sys.argv[1]
    model_path = sys.argv[2] if len(sys.argv) > 2 else '../saved_models/cifar10_cnn_model.h5'
    
    if not os.path.exists(image_path):
        print(f"Error: Image file {image_path} not found!")
        sys.exit(1)
    
    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} not found!")
        print("Please train the model first or specify correct path")
        sys.exit(1)
    
    model = load_model(model_path)
    class_name, confidence = predict_image(model, image_path)
    
    print(f"\nPrediction for {image_path}:")
    print(f"Class: {class_name}")
    print(f"Confidence: {confidence:.2f}%")
