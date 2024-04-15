def train_all_variants(train_model_func, create_model_func, create_data_generator_func, model_variants):
    for variant in model_variants:
        print(f"Training {variant}")
        
        model = create_model_func(variant)  # Use create_model_func to create/load the model
        data_generator = create_data_generator_func(variant)  # Use create_data_generator_func
        
        train_model_func(model=model, data_generator=data_generator, epochs=10, batch_size=64, model_variant=variant)

def evaluate_all_variants(evaluate_model_func, plot_model_metrics_func, tf, model_variants):
    # Evaluate each model variant
    for variant in model_variants:
        print(f"Evaluating {variant}")
        model_path = f"{variant}.h5"
        model = tf.keras.models.load_model(model_path)  # Load model here instead of in evaluation.py
        history, predictions = evaluate_model_func(model_path=model_path, batch_size=32)
        plot_model_metrics_func(history, model_name=variant)
        # Additional evaluation steps...

def predict_with_variants(predict_bounding_box_func, model_variants, image_path='example_image.jpg'):
    # Predict with each model variant using an example image
    for variant in model_variants:
        model_path = f'models/{variant}.h5'  # Construct the model path dynamically
        print(f"Predicting with {variant}")
        predict_bounding_box_func(image_path=image_path, model_path=model_path)  # Pass the model path as an argument
