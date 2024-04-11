import tensorflow as tf
import os
os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=2'
from model import model
from dataloader_eda import data_generator
from model import ShowTestImages
import json
from kerastuner.tuners import RandomSearch
from src.model import MyHyperModel
from tensorflow.keras.preprocessing.image import save_img
from tensorflow.keras.utils import plot_model
import logging
from tqdm import tqdm
from ipywidgets import IntProgress
from IPython.display import display
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def train_model(epochs=1, batch_size=64):  # Adjusted batch_size parameter
    total_images = len(data_generator().dataset)  # Assuming data_generator has a dataset attribute
    steps_per_epoch = total_images // batch_size
    logging.info("Starting model training...")
    progress = IntProgress(min=0, max=epochs)  # epochs should be defined earlier in your code
    display(progress)
    callbacks_list = [
        ShowTestImages(),
        ModelCheckpoint(filepath='model_{epoch:02d}.h5', save_best_only=True, monitor='val_loss'),
        EarlyStopping(monitor='val_loss', patience=3)
    ]
    for epoch in tqdm.tqdm(range(epochs), desc='Epochs'):
        history = model.fit(
            data_generator(),
            epochs=1,  # Running one epoch at a time within tqdm loop
            steps_per_epoch=steps_per_epoch,  # Dynamically calculated steps_per_epoch
            callbacks=callbacks_list
        )
        progress.value += 1  # Update the progress bar
    logging.info("Model training completed.")
    # Save the training history
    with open('model_history.json', 'w') as file:
        json.dump(history.history, file)

model.compile(
    loss={'coords': 'mse'},
    optimizer='sgd',  # Changed from Adam to SGD
    metrics={'coords': 'accuracy'}
)

model.save('/Users/devynmiller/Downloads/ec1-cpsc542/models/car-object-detection.h5')

def tune_model():
    hypermodel = MyHyperModel(input_shape=[380, 676, 3])

    tuner = RandomSearch(
        hypermodel,
        objective='val_accuracy',
        max_trials=10,
        executions_per_trial=2,
        directory='tuner_results',
        project_name='car_object_detection'
    )

    tuner.search_space_summary()

    logging.info("Starting model tuning...")
    tuner.search(data_generator(), epochs=10, validation_split=0.2)
    logging.info("Model tuning completed.")

    best_models = tuner.get_best_models(num_models=5)
    for i, model in enumerate(best_models):
        model.save(f'/Users/devynmiller/Downloads/ec1-cpsc542/models/best_model_{i}.h5')
        plot_model(model, to_file=f'/Users/devynmiller/Downloads/ec1-cpsc542/models/architecture_pngs/model_{i}_architecture.png', show_shapes=True)
