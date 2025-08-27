from data_loading import data_loading
from data_preprocessing import data_preprocessing
from model_generation import model_generation, generate_data, save_generated_data
from evaluation import evaluation
from visualization import create_visualizations


def run_pipeline(dataset: str, model: str, file_path: str = None):
    """Run the complete time series generation pipeline"""

    # Load data
    data = data_loading(dataset, file_path)
    print(
        f"Data loaded successfully: {data.shape if not isinstance(data, tuple) else [d.shape for d in data]}"
    )

    # Preprocess data
    if dataset == "air":
        # Domain adaptation setup returns 4 values
        train_sequences, test_sequences, val_sequences, scaler = data_preprocessing(
            data
        )
        print(f"Data preprocessed successfully (Domain Adaptation)")
        print(f"Train sequences (Tianjin): {train_sequences.shape}")
        print(f"Test sequences (BJ/GZ/SZ): {test_sequences.shape}")
        print(f"Val sequences (Tianjin): {val_sequences.shape}")
    else:
        # Standard setup returns 3 values
        train_sequences, test_sequences, scaler = data_preprocessing(data)
        print(f"Data preprocessed successfully")
        print(f"Train sequences: {train_sequences.shape}")
        print(f"Test sequences: {test_sequences.shape}")

    # Generate model
    model_results = model_generation(train_sequences, test_sequences, model)

    # Add dataset information to model_results
    model_results["dataset"] = dataset

    print(f"Model trained successfully: {model_results['model_type']}")

    # Generate synthetic data
    if model_results.get("is_deep_model", False):
        # Deep learning models already have generated data
        generated_data = model_results["generated_data"]
        print(f"Data generation completed: {generated_data.shape}")
    else:
        # Traditional ML models need data generation
        generated_data = generate_data(model_results)
        print(f"Data generation completed: {generated_data.shape}")

    # Store generated data back in model_results for visualization
    model_results["generated_data"] = generated_data

    # Save generated data
    save_generated_data(generated_data, dataset, model_results["model_type"])

    # Evaluate model
    metrics = evaluation(model_results)

    # Create visualizations
    print("Creating visualizations...")
    create_visualizations(model_results, dataset)

    # Return results
    if dataset == "air":
        return model_results, metrics, generated_data, val_sequences
    else:
        return model_results, metrics, generated_data
