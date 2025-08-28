from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib
import numpy as np
from typing import Dict, Any
import os
import time
import torch
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Deep learning models
from models.timevae import TimeVAE
from models.timegan import TimeGAN
from models.fourier_flows import FourierFlow
from models.time_transformer import create_time_transformer_vae
from config.config import (
    get_model_config,
)


def model_generation(
    train_sequences: np.ndarray,
    test_sequences: np.ndarray,
    model_type: str = "auto",
    max_epochs: int = None,
) -> Dict[str, Any]:
    """Generate and train time series generation models"""
    # All supported models (TimeVAE, TimeGAN, FourierFlow, TimeTransformer, ARIMA, Exp_Smooth)
    # are now handled in _train_deep_model
    return _train_deep_model(train_sequences, test_sequences, model_type, max_epochs)


def _train_deep_model(
    train_sequences: np.ndarray,
    test_sequences: np.ndarray,
    model_type: str,
    max_epochs: int = None,
) -> Dict[str, Any]:
    """Train deep learning models (TimeVAE, TimeGAN, Fourier Flows)"""
    seq_len = train_sequences.shape[1]
    feat_dim = train_sequences.shape[2]

    if model_type == "timevae":
        # Initialize TimeVAE
        config = get_model_config("timevae")
        model = TimeVAE(
            seq_len=seq_len,
            feat_dim=feat_dim,
            latent_dim=config["latent_dim"],
            hidden_layer_sizes=config["hidden_layer_sizes"],
            trend_poly=config["trend_poly"],
            custom_seas=config["custom_seas"],
            use_residual_conn=config["use_residual_conn"],
            reconstruction_wt=config["reconstruction_wt"],
        )

        # Train TimeVAE
        print("Training TimeVAE...")
        start_time = time.time()
        # Use max_epochs if provided, otherwise use config default
        epochs = max_epochs if max_epochs is not None else config["epochs"]
        print(f"Training for {epochs} epochs...")
        model.fit_on_data(train_sequences, max_epochs=epochs, verbose=config["verbose"])
        training_time = time.time() - start_time

        # Generate synthetic data
        generated_data = model.get_prior_samples(len(train_sequences))

    elif model_type == "timegan":
        # Prepare data for TimeGAN (list of sequences)
        data_list = [seq for seq in train_sequences]

        # Initialize TimeGAN
        config = get_model_config("timegan")
        model = TimeGAN(
            seq_len=seq_len,
            feat_dim=feat_dim,
            hidden_dim=config["hidden_dim"],
            num_layers=config["num_layers"],
            z_dim=feat_dim if config["z_dim"] is None else config["z_dim"],
            batch_size=config["batch_size"],
            lr=config["learning_rate"],
        )

        # Train TimeGAN
        print("Training TimeGAN...")
        start_time = time.time()
        # Use max_epochs if provided, otherwise use config default
        iterations = max_epochs if max_epochs is not None else config["epochs"]
        print(f"Training for {iterations} iterations...")
        model.train(data_list, iterations=iterations)
        training_time = time.time() - start_time

        # Generate synthetic data
        generated_data = model.generate(len(train_sequences))
        generated_data = np.array(generated_data)

    elif model_type == "fourierflow":
        # Initialize FourierFlow
        config = get_model_config("fourierflow")
        model = FourierFlow(
            hidden=config["hidden"],
            fft_size=config["fft_size"],
            n_flows=config["n_flows"],
            FFT=config["FFT"],
            flip=config["flip"],
            normalize=config["normalize"],
        )

        # Train FourierFlow
        print("Training FourierFlow...")
        start_time = time.time()
        # Reshape data for FourierFlow: (samples, seq_len, features) -> (samples, seq_len)
        # Use max_epochs if provided, otherwise use config default
        epochs = max_epochs if max_epochs is not None else config["epochs"]
        print(f"Training for {epochs} epochs...")
        model.fit(
            train_sequences,
            epochs=epochs,
            batch_size=config["batch_size"],
            learning_rate=config["learning_rate"],
            display_step=config["display_step"],
        )
        training_time = time.time() - start_time

        # Generate synthetic data
        generated_data = model.sample(len(train_sequences))
        # Reshape back to 3D: (samples, seq_len) -> (samples, seq_len, 1)
        generated_data = generated_data.reshape(-1, seq_len, 1)
        # Repeat for all features
        generated_data = np.repeat(generated_data, feat_dim, axis=2)

        # Update the sequences to use truncated versions
        train_sequences = train_sequences
        test_sequences = test_sequences

    elif model_type == "timetransformer":
        # Initialize TimeTransformer
        config = get_model_config("timetransformer")
        model = create_time_transformer_vae(
            seq_len=seq_len,
            feat_dim=feat_dim,
            latent_dim=config["latent_dim"],
            hidden_layer_sizes=config["hidden_layer_sizes"],
            dilations=config["dilations"],
            k_size=config["k_size"],
            head_size=config["head_size"],
            num_heads=config["num_heads"],
            dropout=config["dropout"],
            reconstruction_wt=config["reconstruction_wt"],
        )

        # Train TimeTransformer
        print("Training TimeTransformer VAE...")
        start_time = time.time()
        # Use max_epochs if provided, otherwise use config default
        epochs = max_epochs if max_epochs is not None else config["epochs"]
        print(f"Training for {epochs} epochs...")
        model.fit(
            train_sequences,
            epochs=epochs,
            batch_size=config["batch_size"],
            learning_rate=config["learning_rate"],
            verbose=True,
        )
        training_time = time.time() - start_time

        # Generate synthetic data
        generated_data = model.sample(len(train_sequences))
        generated_data = generated_data.detach().cpu().numpy()

    elif model_type == "arima":
        # Initialize ARIMA
        config = get_model_config("arima")
        # Use first feature for ARIMA (univariate) - convert to pandas Series
        ts_data = pd.Series(train_sequences[:, :, 0].mean(axis=0))
        model = ARIMA(ts_data, **config)

        # Train ARIMA
        print("Training ARIMA...")
        start_time = time.time()
        model = model.fit()
        training_time = time.time() - start_time

        # Generate synthetic data
        seq_len = train_sequences.shape[1]
        feat_dim = train_sequences.shape[2]
        generated_sequences = []
        for _ in range(len(train_sequences)):
            # Generate future values using ARIMA
            forecast = model.forecast(steps=seq_len)
            sequence = np.tile(forecast, (feat_dim, 1)).T  # Repeat for all features
            generated_sequences.append(sequence)
        generated_data = np.array(generated_sequences)

    elif model_type == "exp_smooth":
        # Initialize Exponential Smoothing
        config = get_model_config("exp_smooth")
        # Use first feature for Exponential Smoothing (univariate) - convert to pandas Series
        ts_data = pd.Series(train_sequences[:, :, 0].mean(axis=0))
        model = ExponentialSmoothing(ts_data, **config)

        # Train Exponential Smoothing
        print("Training Exponential Smoothing...")
        start_time = time.time()
        model = model.fit()
        training_time = time.time() - start_time

        # Generate synthetic data
        seq_len = train_sequences.shape[1]
        feat_dim = train_sequences.shape[2]
        generated_sequences = []
        for _ in range(len(train_sequences)):
            # Generate future values using Exponential Smoothing
            forecast = model.forecast(steps=seq_len)
            sequence = np.tile(forecast, (feat_dim, 1)).T  # Repeat for all features
            generated_sequences.append(sequence)
        generated_data = np.array(generated_sequences)

    else:
        raise ValueError(f"Unsupported deep learning model type: {model_type}")

    # Save model
    if model_type in ["arima", "exp_smooth"]:
        joblib.dump(model, f"model_{model_type}.pkl")
    else:
        # For deep learning models, save to their respective directories
        model_path = f"models/saved_{model_type}"
        os.makedirs(model_path, exist_ok=True)

        if model_type == "timevae":
            model.save(model_path)
        elif model_type == "timegan":
            model.save_model(model_path)
        elif model_type == "timetransformer":
            model.save(model_path)
        elif model_type == "fourierflow":
            # Save PyTorch model
            torch.save(
                model.state_dict(),
                os.path.join(model_path, f"{model_type}_weights.pth"),
            )
            # Save model parameters
            joblib.dump(
                {
                    "hidden": model.hidden if hasattr(model, "hidden") else 64,
                    "fft_size": model.fft_size
                    if hasattr(model, "fft_size")
                    else seq_len,
                    "T": model.d if hasattr(model, "d") else seq_len,
                    "n_flows": len(model.bijectors),
                    "FFT": model.FFT if hasattr(model, "FFT") else False,
                    "flip": model.flips[0] if hasattr(model, "flips") else True,
                    "normalize": model.normalize
                    if hasattr(model, "normalize")
                    else False,
                },
                os.path.join(model_path, f"{model_type}_params.pkl"),
            )

    return {
        "model": model,
        "model_type": model_type,
        "X_test": None,  # Deep learning and time series models don't use X_test/y_test
        "y_test": None,
        "train_sequences": train_sequences,
        "test_sequences": test_sequences,
        "generated_data": generated_data,
        "is_deep_model": True,  # Both deep learning and time series models
        "training_time": training_time,
    }


def generate_data(model_results: Dict[str, Any], num_samples: int = None) -> np.ndarray:
    """Generate synthetic time series data using trained model"""
    if model_results.get("is_deep_model", False):
        # Deep learning models already have generated data
        return model_results["generated_data"]

    # Traditional ML models - use the existing generation logic
    model = model_results["model"]
    train_sequences = model_results["train_sequences"]
    model_type = model_results["model_type"]

    if num_samples is None:
        num_samples = len(train_sequences)

    # Handle time series models differently
    if model_type in ["arima", "exp_smooth"]:
        seq_len = train_sequences.shape[1]
        feat_dim = train_sequences.shape[2]

        # Generate sequences using time series models
        generated_sequences = []
        for _ in range(num_samples):
            if model_type == "arima":
                # Generate future values using ARIMA
                forecast = model.forecast(steps=seq_len)
                sequence = np.tile(forecast, (feat_dim, 1)).T  # Repeat for all features
            else:  # exp_smooth
                # Generate future values using Exponential Smoothing
                forecast = model.forecast(steps=seq_len)
                sequence = np.tile(forecast, (feat_dim, 1)).T  # Repeat for all features
            generated_sequences.append(sequence)

        return np.array(generated_sequences)

    # For traditional ML models (including fallback Linear Regression)
    # These models were trained on flattened sequences, so we need to handle them differently
    seq_len = train_sequences.shape[1]
    feat_dim = train_sequences.shape[2]

    # Generate synthetic sequences by sampling from training distribution
    generated_sequences = []

    for _ in range(num_samples):
        # Sample a random training sequence as starting point
        start_idx = np.random.randint(0, len(train_sequences))
        start_sequence = train_sequences[start_idx].copy()

        # For traditional ML models, we need to use the flattened input format
        # that matches what the model was trained on
        for i in range(len(start_sequence) - 1):
            # Flatten the current sequence up to position i to match training format
            # The model expects flattened sequences (seq_len * feat_dim features)
            current_seq = start_sequence[: i + 1].flatten()

            # Pad or truncate to match the expected input size
            expected_size = seq_len * feat_dim
            if len(current_seq) < expected_size:
                # Pad with zeros if too short
                padded_seq = np.zeros(expected_size)
                padded_seq[: len(current_seq)] = current_seq
                current_seq = padded_seq
            elif len(current_seq) > expected_size:
                # Truncate if too long
                current_seq = current_seq[:expected_size]

            # Reshape to 2D for prediction
            input_seq = current_seq.reshape(1, -1)

            try:
                next_val = model.predict(input_seq)[0]
                # Update the next position in the sequence
                if i + 1 < len(start_sequence):
                    start_sequence[i + 1, 0] = next_val  # Update first feature
            except Exception as e:
                print(f"Warning: Prediction failed at position {i}: {e}")
                # Use the previous value as fallback
                if i > 0:
                    start_sequence[i + 1] = start_sequence[i]

        generated_sequences.append(start_sequence)

    return np.array(generated_sequences)


def save_generated_data(generated_data: np.ndarray, dataset_name: str, model_name: str):
    """Save generated data to file"""
    os.makedirs("data/gen", exist_ok=True)
    output_path = f"data/gen/{dataset_name}_{model_name}_gen.pkl"
    joblib.dump(generated_data, output_path)
    print(f"Generated data saved to {output_path}")
    return output_path
