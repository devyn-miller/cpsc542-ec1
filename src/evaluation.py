import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from dataloader_eda import data_generator  # Assuming this generator can be used for evaluation
from tensorflow.keras import Model
import cv2
import json
import matplotlib.patches as patches
import matplotlib.pyplot as plt


def evaluate():
    # Load the trained model
    model = load_model('car-object-detection.h5')

    # Assuming data_generator is adapted for evaluation (e.g., no data augmentation)
    eval_gen = data_generator(batch_size=32, shuffle=False)  # Adjust parameters as needed

    # Evaluate the model
    results = model.evaluate(eval_gen, steps=50)  # Adjust steps based on your dataset size
    print("Test Loss, Test Accuracy:", results)

    # Evaluate the model on validation data
    val_gen = data_generator(batch_size=32, shuffle=False)  # Make sure this generates validation data
    val_results = model.evaluate(val_gen, steps=50)  # Adjust steps accordingly
    print("Validation Loss, Validation Accuracy:", val_results)

    # Load the training history
    with open('model_history.json', 'r') as file:
        history = json.load(file)
    
    # # Plotting training and validation accuracy
    # plt.plot(history['accuracy'])
    # plt.plot(history['val_accuracy'])
    # plt.title('Model Accuracy')
    # plt.ylabel('Accuracy')
    # plt.xlabel('Epoch')
    # plt.legend(['Train', 'Val'], loc='upper left')
    # plt.show()

    # # Plotting training and validation loss
    # plt.plot(history['loss'])
    # plt.plot(history['val_loss'])
    # plt.title('Model Loss')
    # plt.ylabel('Loss')
    # plt.xlabel('Epoch')
    # plt.legend(['Train', 'Val'], loc='upper left')
    # plt.show()

    # Display a table of metrics
    metrics_table = pd.DataFrame({'Metric': ['Loss', 'Accuracy', 'Validation Loss', 'Validation Accuracy'], 'Value': results + val_results})
    print(metrics_table)

    # Display 3 best and 3 worst results
    # Assuming you have a function to calculate IoU (Intersection over Union) for evaluation
    def calculate_iou(pred_box, true_box):
        """
        Calculate Intersection over Union (IoU) between predicted and true bounding boxes.
        """
        xA = max(pred_box[0], true_box[0])
        yA = max(pred_box[1], true_box[1])
        xB = min(pred_box[2], true_box[2])
        yB = min(pred_box[3], true_box[3])

        # Compute the area of intersection
        intersection = max(0, xB - xA + 1) * max(0, yB - yA + 1)

        # Compute the area of both the prediction and true boxes
        pred_box_area = (pred_box[2] - pred_box[0] + 1) * (pred_box[3] - pred_box[1] + 1)
        true_box_area = (true_box[2] - true_box[0] + 1) * (true_box[3] - true_box[1] + 1)

        # Compute the area of union
        union = pred_box_area + true_box_area - intersection

        # Compute the IoU
        iou = intersection / float(union)
        return iou

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
        """
        Display an image with predicted and true bounding boxes.
        """
        fig, ax = plt.subplots(1)
        ax.imshow(image)

        # True box in green
        rect_true = patches.Rectangle((true_box[0], true_box[1]), true_box[2]-true_box[0], true_box[3]-true_box[1], linewidth=1, edgecolor='g', facecolor='none')
        ax.add_patch(rect_true)

        # Predicted box in red
        rect_pred = patches.Rectangle((pred_box[0], pred_box[1]), pred_box[2]-pred_box[0], pred_box[3]-pred_box[1], linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect_pred)

        plt.show()

    # # Display best 3
    # for iou, image, pred_box, true_box in best_3:
    #     display_with_boxes(image, pred_box, true_box)

    # # Display worst 3
    # for iou, image, pred_box, true_box in worst_3:
    #     display_with_boxes(image, pred_box, true_box)

    # GradCAM or another explainable AI component
    # Assuming you have a function to generate GradCAM visualizations
    def generate_gradcam(model, img_array, layer_name):
        """
        Generate a GradCAM visualization for a specific layer.
        """
        grad_model = Model(inputs=[model.inputs], outputs=[model.get_layer(layer_name).output, model.output])

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(np.array([img_array]))
            loss = predictions[:, np.argmax(predictions[0])]

        output = conv_outputs[0]
        grads = tape.gradient(loss, conv_outputs)[0]

        gate_f = tf.cast(output > 0, 'float32')
        gate_r = tf.cast(grads > 0, 'float32')
        guided_grads = gate_f * gate_r * grads

        weights = tf.reduce_mean(guided_grads, axis=(0, 1))
        cam = np.dot(output, weights[..., np.newaxis])
        cam = np.squeeze(cam)
        cam = cv2.resize(cam, (img_array.shape[1], img_array.shape[0]))
        cam = np.maximum(cam, 0)
        heatmap = (cam - cam.min()) / (cam.max() - cam.min())

        # Convert grayscale heatmap to 3D
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        # Superimpose the heatmap on original image
        superimposed_img = heatmap * 0.4 + img_array
        superimposed_img = np.clip(superimposed_img, 0, 255).astype('uint8')

        return superimposed_img

    # Example usage of GradCAM for a single image
    img = load_img('example_image.jpg', target_size=(224, 224))
    img_array = img_to_array(img)
    def gen_gradcam():
        return generate_gradcam(model, img_array, 'conv2d_9')
    return history, best_3, worst_3, display_with_boxes, gen_gradcam
