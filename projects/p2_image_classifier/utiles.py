import tensorflow as tf
import numpy as np
from PIL import Image

def process_image(image_path):
    image = Image.open(image_path).resize((224, 224))
    image = np.array(image) / 255.0
    return image

def load_model(model_path):
    if model_path.endswith('.h5'):
        return tf.keras.models.load_model(model_path, custom_objects={'KerasLayer': tf.keras.layers.Layer})
    else:
        return tf.keras.models.load_model(model_path)
