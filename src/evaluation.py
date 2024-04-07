import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from dataloader_eda import data_generator  # Assuming this generator can be used for evaluation
from tensorflow.keras import Model
import cv2

# Load the trained model
model = load_model('car-object-detection.h5')

# Assuming data_generator is adapted for evaluation (e.g., no data augmentation)
eval_gen = data_generator(batch_size=32, shuffle=False)  # Adjust parameters as needed

# Evaluate the model
results = model.evaluate(eval_gen, steps=50)  # Adjust steps based on your dataset size
print("Test Loss, Test Accuracy:", results)

# Display a table of metrics
metrics_table = pd.DataFrame({'Metric': ['Loss', 'Accuracy'], 'Value': results})
print(metrics_table)

# Plotting metrics - assuming history object is available from training
# If history object is not saved, you can skip this part or plot metrics from another source
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Display 3 best and 3 worst results
# Assuming you have a function to calculate IoU (Intersection over Union) for evaluation
def calculate_iou(pred_boxes, true_boxes):
    # Implementation of IoU calculation
    pass

# Collect IoUs for a subset of the dataset
ious = []
for images, labels in eval_gen:
    preds = model.predict(images)
    for pred, label in zip(preds, labels['coords']):
        iou = calculate_iou(pred, label)
        ious.append((iou, images, pred, label))

# Sort by IoU and select 3 best and 3 worst
ious.sort(key=lambda x: x[0], reverse=True)
best_3 = ious[:3]
worst_3 = ious[-3:]

# Function to display images with bounding boxes
def display_with_boxes(image, pred_box, true_box):
    # Implementation similar to display_image function in the codebase
    pass

# Display best 3
for iou, image, pred_box, true_box in best_3:
    display_with_boxes(image, pred_box, true_box)

# Display worst 3
for iou, image, pred_box, true_box in worst_3:
    display_with_boxes(image, pred_box, true_box)

# GradCAM or another explainable AI component
# Assuming you have a function to generate GradCAM visualizations
def generate_gradcam(model, img, layer_name='last_conv_layer'):
    # Implementation of GradCAM
    pass

# Example usage of GradCAM for a single image
img = load_img('example_image.jpg', target_size=(224, 224))
img_array = img_to_array(img)
generate_gradcam(model, img_array, 'last_conv_layer_name')
