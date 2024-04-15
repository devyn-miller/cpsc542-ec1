import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array, load_img

def predict_bounding_box(image_path, model_path):
    # Load and preprocess the image
    img = load_img(image_path, target_size=(380, 676))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Model expects 4D tensor

    # Load the model and make predictions
    model = tf.keras.models.load_model(model_path)
    predictions = model.predict(img_array)
    pred_coords = predictions[0]
    print(f"Predictions from {model_path}: {pred_coords}")

# # Example usage
# predict_bounding_box('example_image.jpg', 'models/model_variant_1.h5')
