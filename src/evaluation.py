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
    model = load_model(model_path)
    eval_gen = data_generator(batch_size=batch_size, shuffle=False)
    total_images = len(data_generator().dataset)
    steps = total_images // batch_size
    results = model.evaluate(eval_gen, steps=steps)
    predictions = model.predict(eval_gen, steps=steps)
    print(f"Results for {model_path}: Loss = {results[0]}, Accuracy = {results[1]}")
    history_path = model_path.replace('.h5', '_history.json')
    with open(history_path, 'r') as file:
        history = json.load(file)
    return history, predictions

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

def display_best_worst_results(predictions):
    predictions.sort(key=lambda x: x[0], reverse=True)  # Sort by IoU
    best_3 = predictions[:3]
    worst_3 = predictions[-3:]
    # Display or process best_3 and worst_3 as needed

def generate_gradcam(model, img_array, layer_name):
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
