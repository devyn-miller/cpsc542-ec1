import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, MaxPool2D, Flatten, Dense
import matplotlib.pyplot as plt
from dataloader_eda import data_generator, display_image

def modeling():
    input_ = Input(shape=[380, 676, 3], name='image')

    x = input_

    for i in range(10):
        n_filters = 2**(i + 3)
        x = Conv2D(n_filters, 3, activation='relu', padding='same')(x)
        x = BatchNormalization()(x)
        x = MaxPool2D(2, padding='same')(x)

    x = Flatten()(x)
    x = Dense(256, activation='relu')(x)
    x = Dense(32, activation='relu')(x)
    output = Dense(4, name='coords')(x)

    model = tf.keras.models.Model(input_, output)
    model.summary()

    model.compile(
        loss={
            'coords': 'mse'
        },
        optimizer=tf.keras.optimizers.Adam(1e-3),
        metrics={
            'coords': 'accuracy'
        }
    )


    # Some functions to test the model. These will be called every epoch to display the current performance of the model
    def test_model(model, datagen):
        example, label = next(datagen)
        
        X = example['image']
        y = label['coords']
        
        pred_bbox = model.predict(X)[0]
        
        img = X[0]
        gt_coords = y[0]
        
        display_image(img, pred_coords=pred_bbox, norm=True)

    def test(model):
        datagen = data_generator(batch_size=1)
        
        plt.figure(figsize=(15,7))
        for i in range(3):
            plt.subplot(1, 3, i + 1)
            test_model(model, datagen)    
        plt.show()
        
    class ShowTestImages(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            test(self.model)

    return model, test(model)