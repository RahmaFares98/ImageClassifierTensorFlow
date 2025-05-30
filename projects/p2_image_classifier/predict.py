import argparse
import tensorflow as tf
import numpy as np
import json
from utils import process_image, load_model

def predict(image_path, model_path, top_k=5, category_names=None):
    model = load_model(model_path)
    image = process_image(image_path)
    image = np.expand_dims(image, axis=0)

    predictions = model.predict(image)[0]
    top_indices = predictions.argsort()[-top_k:][::-1]
    top_probs = predictions[top_indices]
    top_classes = top_indices + 1  # classes start at 1, not 0

    if category_names:
        with open(category_names, 'r') as f:
            class_names = json.load(f)
        top_labels = [class_names.get(str(cls), f"Class {cls}") for cls in top_classes]
    else:
        top_labels = [str(cls) for cls in top_classes]

    return top_probs, top_classes, top_labels

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict flower name from image and model.')
    parser.add_argument('image_path', help='Path to input image.')
    parser.add_argument('model', help='Path to trained model (.h5 or SavedModel).')
    parser.add_argument('--top_k', type=int, default=5, help='Return top K most likely classes.')
    parser.add_argument('--category_names', help='Path to JSON file mapping labels to flower names.')

    args = parser.parse_args()
    probs, classes, labels = predict(args.image_path, args.model, args.top_k, args.category_names)

    print("\nPrediction Results:")
    for i in range(len(probs)):
        print(f"{labels[i]}: {probs[i]*100:.2f}%")
