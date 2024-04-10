import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, MaxPool2D, Flatten, Dense, Activation, GlobalAveragePooling2D, MaxPooling2D
import matplotlib.pyplot as plt
# from dataloader_eda import data_generator, display_image
from kerastuner import HyperModel
from kerastuner.tuners import RandomSearch
from tqdm import tqdm
import logging
from tensorflow.keras.callbacks import Callback
from tqdm.notebook import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class MyHyperModel(HyperModel):
    def __init__(self, input_shape):
        self.input_shape = input_shape

    def build(self, hp):
        input_ = Input(shape=self.input_shape, name='image')
        x = input_
        for i in range(hp.Int('conv_blocks', 3, 4, default=3)):  # Reduced max value
            filters = hp.Int('filters_' + str(i), 32, 128, step=32)  # Narrowed range
            for _ in range(2):
                x = Conv2D(filters, kernel_size=(3, 3), padding='same')(x)
                x = BatchNormalization()(x)
                x = Activation('relu')(x)
            if i < hp.get('conv_blocks') - 1:
                x = MaxPooling2D(pool_size=(2, 2))(x)
        x = GlobalAveragePooling2D()(x)
        x = Dense(hp.Int('dense_units', 32, 128, step=32), activation='relu')(x)
        output = Dense(4, activation='linear', name='coords')(x)
        model = tf.keras.Model(inputs=input_, outputs=output)

        model.compile(optimizer=tf.keras.optimizers.Adam(hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4])),
                      loss='mse',
                      metrics=['accuracy'])
        return model

class ShowTestImages(tf.keras.callbacks.Callback):
    def __init__(self, test_func):
        super(ShowTestImages, self).__init__()
        self.test_func = test_func

    def on_epoch_end(self, epoch, logs=None):
        self.test_func(self.model)

class TQDMNotebookCallback(Callback):
    def on_train_begin(self, logs=None):
        self.epochs = self.params['epochs']
        self.progress_bar = tqdm(total=self.epochs)
        self.progress_bar.set_description('Training Progress')

    def on_epoch_end(self, epoch, logs=None):
        self.progress_bar.update(1)

from kerastuner.engine import tuner_utils
from tensorflow.keras.callbacks import Callback

class TQDMTunerCallback(Callback):
    def __init__(self, total_trials, **kwargs):
        super().__init__(**kwargs)
        self.total_trials = total_trials
        self.progress_bar = tqdm(total=total_trials, desc='Tuning Progress')

    def on_trial_end(self, trial, logs=None):
        self.progress_bar.update(1)

def tune_model(data_generator, display_image, df, path, epochs):
    hypermodel = MyHyperModel(input_shape=[380, 676, 3])

    def test_model(model, datagen, title=""):
        example, label = next(datagen)
        
        X = example['image']
        y = label['coords']
        
        pred_bbox = model.predict(X)[0]
        
        img = X[0]
        gt_coords = y[0]
        
        display_image(img, pred_coords=pred_bbox, norm=True)
        plt.title(title)

    def test(model):
        datagen = data_generator(batch_size=1)
        
        # Define the grid size
        grid_size = (3, 3)  # for example, a 3x3 grid
        plt.figure(figsize=(15,15))  # Adjust the figure size as needed
        
        for i in range(grid_size[0] * grid_size[1]):
            plt.subplot(grid_size[0], grid_size[1], i + 1)
            test_model(model, datagen, title=f"Image {i+1}")
        
        plt.tight_layout()  # This will make the layout organized and nicely spaced
        plt.show()
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
    # Assuming df and path are defined and valid
    tuner.search(data_generator(df=df, batch_size=32, path=path), epochs=epochs, callbacks=[ShowTestImages(test), TQDMTunerCallback(total_trials=10)], verbose=0)
    logging.info("Model tuning completed.")

    best_model = tuner.get_best_models(num_models=1)[0]
    logging.info("Saving the best model as 'best_model.h5'.")
    best_model.save('best_model.h5')

    # Some functions to test the model. These will be called every epoch to display the current performance of the model
    def test_model(model, datagen, title=""):
        example, label = next(datagen)
        
        X = example['image']
        y = label['coords']
        
        pred_bbox = model.predict(X)[0]
        
        img = X[0]
        gt_coords = y[0]
        
        display_image(img, pred_coords=pred_bbox, norm=True)
        plt.title(title)

    def test(model):
        datagen = data_generator(batch_size=1)
        
        plt.figure(figsize=(15,7))
        for i in range(3):
            plt.subplot(1, 3, i + 1)
            test_model(model, datagen)    
        plt.show()
        
    return best_model
