import numpy as np
import tensorflow as tf
from dataloader_eda import load_data
from tensorflow.keras.preprocessing.image import img_to_array, load_img

# Load the trained model
model = tf.keras.models.load_model('car-object-detection.h5')

def predict_bounding_box(image_path):
    """
    Predicts the bounding box coordinates for a given image.

    Parameters:
    - image_path: Path to the image file.

    Returns:
    - pred_coords: Predicted bounding box coordinates (xmin, ymin, xmax, ymax).
    """
    # Load and preprocess the image
    _, train_path, _ = load_data()  # Assuming the image is in the training path
    img = load_img(train_path / image_path, target_size=(380, 676))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # Model expects 4D tensor
    predictions = model.predict(img_array)
    # Assuming the model returns coordinates in the format [xmin, ymin, xmax, ymax]
    pred_coords = predictions[0]
    return pred_coords
