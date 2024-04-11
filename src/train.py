import tensorflow as tf
import os
os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=2'
from model import model
from dataloader_eda import data_generator
# from model import ShowTestImages
import json
import logging
# from tqdm import tqdm
from ipywidgets import IntProgress
from IPython.display import display
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def train_model(epochs=1, batch_size=64, model_variant=''):  # Adjusted batch_size parameter and added model_variant
    total_images = len(data_generator().dataset)  # Assuming data_generator has a dataset attribute
    steps_per_epoch = total_images // batch_size
    logging.info("Starting model training for " + model_variant)
    callbacks_list = [
        ModelCheckpoint(filepath=model_variant + '_{epoch:02d}.h5', save_best_only=True, monitor='val_loss'),
        EarlyStopping(monitor='val_loss', patience=3)
    ]
    for epoch in range(epochs):
        history = model.fit(
            data_generator(),
            epochs=1,  # Running one epoch at a time
            steps_per_epoch=steps_per_epoch,  # Dynamically calculated steps_per_epoch
            callbacks=callbacks_list,
            verbose=1  # Ensures the built-in Keras progress bar is displayed
        )
    logging.info("Model training completed for " + model_variant)
    # Save the training history
    with open(model_variant + '_history.json', 'w') as file:
        json.dump(history.history, file)

# Example of saving models after training with specific modifications
# Assuming the training function is called with the appropriate model_variant argument
train_model(epochs=10, batch_size=64, model_variant='model_variant_1')
train_model(epochs=10, batch_size=64, model_variant='model_variant_2')
train_model(epochs=10, batch_size=64, model_variant='model_variant_3')
train_model(epochs=10, batch_size=64, model_variant='model_variant_4')
train_model(epochs=10, batch_size=64, model_variant='model_variant_5')

model.compile(
    loss={'coords': 'mse'},
    optimizer='sgd',  # Changed from Adam to SGD
    metrics={'coords': 'accuracy'}
)

model.save('/Users/devynmiller/Downloads/ec1-cpsc542/models/car-object-detection.h5')

