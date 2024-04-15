from tensorflow.keras.preprocessing.image import ImageDataGenerator

def create_augmentation():
    # Define your augmentation logic here
    return ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

# # example use
# augmentation = create_augmentation()
