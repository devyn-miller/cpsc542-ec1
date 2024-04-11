from src.train import train_model
from src.evaluation import evaluate_model, plot_model_metrics, display_best_worst_results, generate_gradcam
from src.predict import predict_bounding_box
from src.dataloader_eda import data_generator, load_data
from src.augmentation import create_augmentation

# Define model variants
model_variants = ['model_variant_1', 'model_variant_2', 'model_variant_3', 'model_variant_4', 'model_variant_5']

# Preprocessing and Augmentation
# Assuming data_generator and load_data are set up for both training and validation data
# and create_augmentation for training data augmentation

# Train each model variant
for variant in model_variants:
    print(f"Training {variant}")
    train_model(epochs=10, batch_size=64, model_variant=variant)

# Evaluate each model variant and display metrics, best/worst results, and GradCAM
for variant in model_variants:
    print(f"Evaluating {variant}")
    model_path = f"{variant}.h5"
    history, predictions = evaluate_model(model_path=model_path, batch_size=32)
    plot_model_metrics(history, model_name=variant)
    
    # Display best and worst predictions
    display_best_worst_results(predictions)
    
    # Generate and display GradCAM visualizations for selected images
    generate_gradcam(model_path, selected_images=['example_image.jpg'])

# Predict with each model variant using an example image
image_path = 'example_image.jpg'  # Ensure this image is available in your dataset
for variant in model_variants:
    print(f"Predicting with {variant}")
    predict_bounding_box(image_path=image_path)
