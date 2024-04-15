import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Dense, Activation, GlobalAveragePooling2D, MaxPooling2D, Dropout
import matplotlib.pyplot as plt
# from dataloader_eda import data_generator, display_image
import logging
# from tensorflow.keras.callbacks import Callback
import copy
from keras_tuner import HyperModel

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class MyHyperModel(HyperModel):
    def __init__(self, input_shape):
        self.input_shape = input_shape

    def build(self, hp):
        input_ = Input(shape=self.input_shape, name='image')
        x = input_
        # Model 1: Doubling the number of filters in the first Conv2D layer
        x = Conv2D(64, kernel_size=(3, 3), padding='same')(x)  # Change for Model 1
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(64, kernel_size=(3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('elu')(x)  # Change for Model 2
        x = MaxPooling2D(pool_size=(2, 2))(x)
        # Model 4: Adding a dropout layer
        x = Dropout(0.2)(x)  # Adding a dropout layer with a 20% dropout rate
        x = GlobalAveragePooling2D()(x)
        x = Dense(64, activation='relu')(x)
        output = Dense(4, activation='linear', name='coords')(x)
        model = tf.keras.Model(inputs=input_, outputs=output)

        # Model 3: Using RMSprop instead of Adam
        model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=1e-3),
                      loss='mse',
                      metrics=['accuracy'])
        return model

def train_and_evaluate_model(model, data_generator, display_image, df, path, epochs):
    # Assuming df and path are defined and valid
    model.fit(data_generator(df=df, batch_size=32, path=path), epochs=epochs)
    logging.info("Model training completed.")

    logging.info("Evaluating the model.")
    # Evaluation logic here

    logging.info("Saving the model.")
    model.save('model_variant.h5')

# Model 5: Adjusting Batch Normalization placement
class MyAdjustedHyperModel(HyperModel):
    def __init__(self, input_shape):
        self.input_shape = input_shape

    def build(self, hp):
        input_ = Input(shape=self.input_shape, name='image')
        x = input_
        x = BatchNormalization()(x)  # Moving BatchNormalization before Conv2D
        x = Conv2D(32, kernel_size=(3, 3), padding='same')(x)
        x = Activation('relu')(x)
        x = Conv2D(32, kernel_size=(3, 3), padding='same')(x)
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
        x = GlobalAveragePooling2D()(x)
        x = Dense(32, activation='relu')(x)
        output = Dense(4, activation='linear', name='coords')(x)
        model = tf.keras.Model(inputs=input_, outputs=output)

        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                      loss='mse',
                      metrics=['accuracy'])
        return model

# At the end of src/model.py
input_shape = (224, 224, 3)  # Adjust this as per your actual input shape
hypermodel = MyAdjustedHyperModel(input_shape=input_shape)
def model():
    return hypermodel.build(hp=None)  # Assuming hp (HyperParameters) is not needed for this simple example

# # Display model summary to verify the structure
# model.summary()
