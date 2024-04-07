import tensorflow as tf
from model import model
from dataloader_eda import data_generator
from model import ShowTestImages
import json

def train_model():
    with tf.device('/GPU:0'):
        history = model.fit(
            data_generator(),
            epochs=1,
            steps_per_epoch=5,
            callbacks=[
                ShowTestImages(),
            ]
        )
    # Save the training history
    with open('model_history.json', 'w') as file:
        json.dump(history.history, file)

# model.save('car-object-detection.h5')