from src.augmentation import augment
import tensorflow as tf
def predict(model, test_batches):
    train_batches, test_batches, display, sample_image, sample_mask = augment()
    """Try out the model to check what it predicts before training:"""

    def create_mask(pred_mask):
        # Create a mask from the predicted mask
        pred_mask = tf.math.argmax(pred_mask, axis=-1)
        pred_mask = pred_mask[..., tf.newaxis]
        return pred_mask[0]

    def show_predictions(dataset=None, num=1):
        # Show the predictions
        if dataset:
            for image, mask in dataset.take(num):
                pred_mask = model.predict(image)
                display([image[0], mask[0], create_mask(pred_mask)])
        else:
            display([sample_image, sample_mask,
                    create_mask(model.predict(sample_image[tf.newaxis, ...]))])

    show_predictions(test_batches, 3)
    return show_predictions