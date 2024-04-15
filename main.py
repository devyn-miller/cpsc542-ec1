import warnings
warnings.filterwarnings("ignore")
import importlib
import tensorflow as tf
import os

# Import modules from src
import src.augmentation as augmentation_module
import src.train as train_module
import src.evaluation as evaluation_module
import src.predict as predict_module
import src.dataloader_eda as dataloader_eda_module
import src.model as model_module  # Import the model

# Reload the modules to ensure any changes are applied
augmentation = importlib.reload(augmentation_module)
train = importlib.reload(train_module)
evaluation = importlib.reload(evaluation_module)
predict = importlib.reload(predict_module)
dataloader_eda = importlib.reload(dataloader_eda_module)
model = importlib.reload(model_module)

# Initialize data augmentation
augmentation_instance = augmentation.create_augmentation()

# Define model variants
model_variants = ['model_variant_1', 'model_variant_2', 'model_variant_3', 'model_variant_4', 'model_variant_5']

# Load data and define train_df and train_path
_, train_path, train_df = dataloader_eda.load_data()

# Train all model variants with augmentation
evaluation.train_all_variants(
    train.train_model,
    lambda variant: model.model(),  # Assuming this correctly creates a model based on the variant
    lambda variant: lambda: dataloader_eda.data_generator(df=train_df, batch_size=64, path=train_path, augmentation=augmentation_instance),
    model_variants
)

# Assuming the rest of the code for prediction remains the same
training_image_directory = 'data/training_images/'
testing_image_directory = 'data/testing_images/'

training_image_filenames = [os.path.join(training_image_directory, filename) for filename in os.listdir(training_image_directory) if filename.endswith('.jpg')]
testing_image_filenames = [os.path.join(testing_image_directory, filename) for filename in os.listdir(testing_image_directory) if filename.endswith('.jpg')]

# Combine both lists
image_filenames = training_image_filenames + testing_image_filenames

# Use the first image for prediction as an example
first_image_path = image_filenames[0] if image_filenames else None

if first_image_path:
    evaluation.predict_with_variants(predict.predict_bounding_box, model_variants, first_image_path)
else:
    print("No images found for prediction.")

# Predict on all images
for image_path in image_filenames:
    evaluation.predict_with_variants(predict.predict_bounding_box, model_variants, image_path)
