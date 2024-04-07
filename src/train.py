import tensorflow as tf
from model import model
from dataloader_eda import data_generator
from model import ShowTestImages

with tf.device('/GPU:0'):
    _ = model.fit(
        data_generator(),
        epochs=1,
        steps_per_epoch=5,
        callbacks=[
            ShowTestImages(),
        ]
    )

# model.save('car-object-detection.h5')