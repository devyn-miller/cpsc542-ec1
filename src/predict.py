import numpy as np
import tensorflow as tf
from dataloader_eda import load_data
from tensorflow.keras.preprocessing.image import img_to_array, load_img

# List of model variants
model_paths = [
    'model_variant_1.h5',
    'model_variant_2.h5',
    'model_variant_3.h5',
    'model_variant_4.h5',
    'model_variant_5.h5'
]

def predict_bounding_box(image_path):
    # Load and preprocess the image
    _, train_path, _ = load_data()
    img = load_img(train_path / image_path, target_size=(380, 676))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Model expects 4D tensor

    # Iterate over each model variant and make predictions
    for model_path in model_paths:
        model = tf.keras.models.load_model(model_path)
        predictions = model.predict(img_array)
        pred_coords = predictions[0]
        print(f"Predictions from {model_path}: {pred_coords}")

# Example usage
predict_bounding_box('example_image.jpg')
