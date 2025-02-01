import argparse
import tensorflow as tf
import numpy as np
from PIL import Image
import json

def process_image(image_path):
    """Preprocess image: Resize, normalize, and convert to NumPy array."""
    image = Image.open(image_path)
    image = image.resize((224, 224))  
    image = np.asarray(image) / 255.0 
    return np.expand_dims(image, axis=0)  

def predict(image_path, model, top_k=5):
    """Predict top K most likely classes for an image."""
    processed_image = process_image(image_path)
    predictions = model.predict(processed_image)[0]
    
    top_k_indices = np.argsort(predictions)[-top_k:][::-1] 
    top_k_probs = predictions[top_k_indices]  

    return top_k_probs, top_k_indices

def main():
    parser = argparse.ArgumentParser(description="Predict flower name from an image using a trained deep learning model.")
    
    parser.add_argument("image_path", type=str, help="Path to input image")
    parser.add_argument("model_path", type=str, help="Path to saved Keras model")
    parser.add_argument("--top_k", type=int, default=5, help="Return the top K most likely classes")
    parser.add_argument("--category_names", type=str, help="Path to JSON file mapping labels to flower names")
    
    args = parser.parse_args()

    # Load model
    model = tf.keras.models.load_model(args.model_path)
    
    # Predict
    probs, classes = predict(args.image_path, model, args.top_k)
    
    # Load category names if provided
    if args.category_names:
        with open(args.category_names, "r") as f:
            class_names = json.load(f)
        classes = [class_names[str(cls)] for cls in classes]  # Convert indices to flower names

    # Print results
    print("\nPredictions:")
    for i in range(len(classes)):
        print(f"{classes[i]}: {probs[i]:.3f}")

if __name__ == "__main__":
    main()
