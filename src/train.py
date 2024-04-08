import tensorflow as tf
from model import model
from dataloader_eda import data_generator
from model import ShowTestImages
import json
from kerastuner.tuners import RandomSearch
from src.model import MyHyperModel

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

model.save('car-object-detection.h5')

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

    tuner.search(data_generator(), epochs=10, validation_split=0.2)

    best_models = tuner.get_best_models(num_models=5)
    for i, model in enumerate(best_models):
        model.save(f'best_model_{i}.h5')
