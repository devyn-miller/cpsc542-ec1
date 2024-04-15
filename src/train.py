import os
import json
import logging
from tqdm import tqdm
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def train_model(model, data_generator, epochs=1, batch_size=64, model_variant='', augmentation=None):
    total_images = len(data_generator().dataset)  # Assuming data_generator has a dataset attribute
    steps_per_epoch = total_images // batch_size
    logging.info("Starting model training for " + model_variant)
    callbacks_list = [
        ModelCheckpoint(filepath='models/' + model_variant + '_{epoch:02d}.h5', save_best_only=True, monitor='val_loss'),
        EarlyStopping(monitor='val_loss', patience=3)
    ]
    for epoch in tqdm(range(epochs), desc="Training Progress"):
        history = model.fit(
            data_generator(),
            epochs=1,  # Running one epoch at a time
            steps_per_epoch=steps_per_epoch,  # Dynamically calculated steps_per_epoch
            callbacks=callbacks_list,
            verbose=0  # Disables the built-in Keras progress bar
        )
    logging.info("Model training completed for " + model_variant)
    # Save the training history
    with open('models/' + model_variant + '_history.json', 'w') as file:
        json.dump(history.history, file)
    # Save the model with a distinct name
    model.save('models/' + model_variant + '.h5')
