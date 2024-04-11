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
import os
from tensorflow.keras.preprocessing.image import save_img

def evaluate_model(model_path, batch_size=32):  # Add batch_size parameter
    total_images = len(data_generator().dataset)  # Assuming data_generator has a dataset attribute
    steps = total_images // batch_size
    
    model = load_model(model_path)
    eval_gen = data_generator(batch_size=batch_size, shuffle=False)
    results = model.evaluate(eval_gen, steps=steps)
    print(f"Results for {model_path}: Loss = {results[0]}, Accuracy = {results[1]}")
    # Assuming the model's training history is saved in a JSON file named similarly to the model
    history_path = model_path.replace('.h5', '_history.json')
    with open(history_path, 'r') as file:
        history = json.load(file)
    return history

def plot_model_metrics(history, model_name):
    # Plotting training and validation accuracy
    plt.plot(history['accuracy'])
    plt.plot(history['val_accuracy'])
    plt.title(f'Model Accuracy for {model_name}')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.show()

    # Plotting training and validation loss
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title(f'Model Loss for {model_name}')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc='upper left')
    plt.show()

def evaluate():
    # Load the trained model
    model_paths = ['model_variant_1.h5', 'model_variant_2.h5', 'model_variant_3.h5', 'model_variant_4.h5', 'model_variant_5.h5']
    for model_path in model_paths:
        model = load_model(model_path)

        # Assuming data_generator is adapted for evaluation (e.g., no data augmentation)
        eval_gen = data_generator(batch_size=32, shuffle=False)  # Adjust parameters as needed

        # Evaluate the model
        total_images = len(data_generator().dataset)  # Assuming data_generator has a dataset attribute
        steps = total_images // 32
        results = model.evaluate(eval_gen, steps=steps)  # Adjust steps based on your dataset size
        print(f"Test Loss, Test Accuracy for {model_path}:", results)

        # Evaluate the model on validation data
        val_gen = data_generator(batch_size=32, shuffle=False)  # Make sure this generates validation data
        val_results = model.evaluate(val_gen, steps=steps)  # Adjust steps accordingly
        print(f"Validation Loss, Validation Accuracy for {model_path}:", val_results)

        # Load the training history
        history_path = model_path.replace('.h5', '_history.json')
        with open(history_path, 'r') as file:
            history = json.load(file)
        
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

        def display_with_boxes(image, pred_box, true_box, save_path):
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

            plt.savefig(save_path)
            plt.close()

        # GradCAM or another explainable AI component
        # Assuming you have a function to generate GradCAM visualizations
        def generate_gradcam(model_path, img_array, layer_name):
            model = load_model(model_path)
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

            gradcam_save_path = f'/Users/devynmiller/Downloads/ec1-cpsc542/PNGs/gradcam/gradcam_image.png'
            save_img(gradcam_save_path, superimposed_img)

            return superimposed_img

        # Example usage of GradCAM for a single image
        img = load_img('example_image.jpg', target_size=(224, 224))
        img_array = img_to_array(img)
        def gen_gradcam():
            return generate_gradcam(model, img_array, 'conv2d_9')
        return history, best_3, worst_3, display_with_boxes, gen_gradcam
