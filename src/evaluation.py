import os
import time
import json
import numpy as np
import torch
import torch.nn as nn
from scipy.linalg import sqrtm
from sklearn.metrics import mean_squared_error, mean_absolute_error
from typing import Dict, Any, Tuple

from src.model_generation import generate_data as gd
import warnings

warnings.filterwarnings("ignore")


def evaluation(model_results: Dict[str, Any]) -> Dict[str, float]:
    """Evaluate model performance with comprehensive metrics"""
    model = model_results["model"]
    train_sequences = model_results["train_sequences"]
    test_sequences = model_results["test_sequences"]

    # Generate synthetic data if not already generated
    if model_results.get("is_deep_model", False):
        generated_data = model_results["generated_data"]
    else:
        generated_data = generate_data(model_results)

    print(f"Evaluating {model_results['model_type']} model...")

    # Basic metrics
    metrics = {}

    # RMSE and MAE - calculate for all models
    if model_results.get("is_deep_model", False):
        # For deep learning models, calculate reconstruction error
        # Use test sequences as ground truth and generated data as prediction

        # Ensure we have the same number of samples for comparison
        min_samples = min(len(test_sequences), len(generated_data))
        test_subset = test_sequences[:min_samples]
        gen_subset = generated_data[:min_samples]

        # Ensure same sequence length and feature dimensions
        min_seq_len = min(test_subset.shape[1], gen_subset.shape[1])
        min_feat = min(test_subset.shape[2], gen_subset.shape[2])

        test_subset = test_subset[:, :min_seq_len, :min_feat]
        gen_subset = gen_subset[:, :min_seq_len, :min_feat]

        # Flatten for metric calculation
        test_flat = test_subset.reshape(-1)
        gen_flat = gen_subset.reshape(-1)

        # Ensure same length
        min_length = min(len(test_flat), len(gen_flat))
        test_flat = test_flat[:min_length]
        gen_flat = gen_flat[:min_length]

        print(
            f"  Comparing {min_samples} samples, {min_seq_len} seq_len, {min_feat} features"
        )
        print(f"  Flattened lengths: test={len(test_flat)}, generated={len(gen_flat)}")

        metrics["RMSE"] = np.sqrt(mean_squared_error(test_flat, gen_flat))
        metrics["MAE"] = mean_absolute_error(test_flat, gen_flat)
    else:
        # Traditional ML models
        X_test = model_results["X_test"]
        y_test = model_results["y_test"]

        # Handle time series models differently
        if model_results.get("model_type") in ["arima", "exp_smooth"]:
            # For time series models, calculate RMSE and MAE using generated vs test sequences

            generated_data = gd(model_results, num_samples=len(test_sequences))

            # Flatten sequences for comparison
            test_flat = test_sequences.reshape(-1)
            gen_flat = generated_data.reshape(-1)

            # Ensure same length
            min_len = min(len(test_flat), len(gen_flat))
            test_flat = test_flat[:min_len]
            gen_flat = gen_flat[:min_len]

            metrics["RMSE"] = np.sqrt(mean_squared_error(test_flat, gen_flat))
            metrics["MAE"] = mean_absolute_error(test_flat, gen_flat)
        else:
            # Regular ML models
            y_pred = model.predict(X_test)
            metrics["RMSE"] = np.sqrt(mean_squared_error(y_test, y_pred))
            metrics["MAE"] = mean_absolute_error(y_test, y_pred)

    # Training time - get from model results if available
    if "training_time" in model_results:
        metrics["Training_Time"] = model_results["training_time"]
    else:
        metrics["Training_Time"] = 0.0  # Placeholder

    # Ensure data has same shape for comparison
    if generated_data.shape != train_sequences.shape:
        # Pad or truncate to match shapes
        min_samples = min(generated_data.shape[0], train_sequences.shape[0])
        min_seq_len = min(generated_data.shape[1], train_sequences.shape[1])
        min_feat = min(generated_data.shape[2], train_sequences.shape[2])

        ori_data = train_sequences[:min_samples, :min_seq_len, :min_feat]
        gen_data = generated_data[:min_samples, :min_seq_len, :min_feat]
    else:
        ori_data = train_sequences
        gen_data = generated_data

    # Calculate all metrics
    metrics.update(calculate_comprehensive_metrics(ori_data, gen_data))

    # Print results
    print(f"\nEvaluation Results for {model_results['model_type']}:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

    # Save results to JSON file
    # Extract dataset from model_results if available, otherwise use a default
    dataset = model_results.get("dataset", "unknown")
    save_results_to_json(model_results["model_type"], metrics, dataset, generated_data)

    return metrics


def save_results_to_json(
    model_type: str,
    metrics: Dict[str, float],
    dataset: str,
    generated_data: np.ndarray = None,
) -> None:
    """Save evaluation results to a JSON file

    The JSON structure will be:
    {
      "google": {
        "_generated_data_info": {
          "num_series": 100,
          "length_per_series": 125,
          "num_features": 5
        },
        "arima": {
          "RMSE": 0.1234,
          "MAE": 0.0987,
          "Training_Time": 45.67,
          ...
        },
        "timevae": {
          "RMSE": 0.2345,
          "MAE": 0.1876,
          "Training_Time": 120.45,
          ...
        }
      },
      "air": {
        "_generated_data_info": {
          "num_series": 80,
          "length_per_series": 100,
          "num_features": 6
        },
        "arima": {
          ...
        }
      }
    }
    """
    results_file = "results/results.json"

    # Load existing results if file exists
    existing_results = {}
    if os.path.exists(results_file):
        try:
            with open(results_file, "r") as f:
                existing_results = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            existing_results = {}

    # Initialize dataset structure if it doesn't exist
    if dataset not in existing_results:
        existing_results[dataset] = {}

    # Update dataset-level metadata with dynamic generated data info
    if generated_data is not None and len(generated_data.shape) == 3:
        if "_generated_data_info" not in existing_results[dataset]:
            existing_results[dataset]["_generated_data_info"] = {}
        existing_results[dataset]["_generated_data_info"].update(
            {
                "num_series": int(generated_data.shape[0]),
                "length_per_series": int(generated_data.shape[1]),
                "num_features": int(generated_data.shape[2]),
            }
        )

    # Add new results under the dataset
    existing_results[dataset][model_type] = metrics

    # Save updated results
    try:
        with open(results_file, "w") as f:
            json.dump(existing_results, f, indent=2)
        print(
            f"Results saved to {results_file} for dataset: {dataset}, model: {model_type}"
        )
    except Exception as e:
        print(f"Error saving results to JSON: {e}")


def calculate_comprehensive_metrics(
    ori_data: np.ndarray, gen_data: np.ndarray
) -> Dict[str, float]:
    """Calculate comprehensive evaluation metrics"""
    metrics = {}

    # Distance-based measures
    metrics["ED"] = calculate_euclidean_distance(ori_data, gen_data)
    metrics["DTW"] = calculate_dtw_distance(ori_data, gen_data)

    # Feature-based measures
    metrics["MDD"] = calculate_marginal_distribution_difference(ori_data, gen_data)
    metrics["ACD"] = calculate_autocorrelation_difference(ori_data, gen_data)
    metrics["SD"] = calculate_skewness_difference(ori_data, gen_data)
    metrics["KD"] = calculate_kurtosis_difference(ori_data, gen_data)

    # Model-based measures (simplified versions)
    metrics["DS"] = calculate_discriminative_score(ori_data, gen_data)
    metrics["PS"] = calculate_predictive_score(ori_data, gen_data)

    # Contextual-FID (simplified)
    metrics["C-FID"] = calculate_contextual_fid(ori_data, gen_data)

    return metrics


def calculate_euclidean_distance(ori_data: np.ndarray, gen_data: np.ndarray) -> float:
    """Calculate average Euclidean distance between original and generated data"""
    distances = []
    for i in range(ori_data.shape[0]):
        total_dist = 0
        for j in range(ori_data.shape[2]):  # features
            dist = np.linalg.norm(ori_data[i, :, j] - gen_data[i, :, j])
            total_dist += dist
        distances.append(total_dist / ori_data.shape[2])
    return np.mean(distances)


def calculate_dtw_distance(ori_data: np.ndarray, gen_data: np.ndarray) -> float:
    """Calculate average DTW distance (simplified implementation)"""
    distances = []
    for i in range(min(ori_data.shape[0], 10)):  # Limit samples for speed
        # Simple DTW approximation
        seq1 = ori_data[i].flatten()
        seq2 = gen_data[i].flatten()
        dist = np.sum(np.abs(seq1 - seq2))  # Simplified DTW
        distances.append(dist)
    return np.mean(distances)


def calculate_marginal_distribution_difference(
    ori_data: np.ndarray, gen_data: np.ndarray
) -> float:
    """Calculate marginal distribution difference using histogram comparison"""
    differences = []
    for feat in range(ori_data.shape[2]):
        ori_flat = ori_data[:, :, feat].flatten()
        gen_flat = gen_data[:, :, feat].flatten()

        # Create histograms
        bins = np.linspace(
            min(ori_flat.min(), gen_flat.min()), max(ori_flat.max(), gen_flat.max()), 50
        )
        ori_hist, _ = np.histogram(ori_flat, bins=bins, density=True)
        gen_hist, _ = np.histogram(gen_flat, bins=bins, density=True)

        # Calculate difference
        diff = np.mean(np.abs(ori_hist - gen_hist))
        differences.append(diff)

    return np.mean(differences)


def calculate_autocorrelation_difference(
    ori_data: np.ndarray, gen_data: np.ndarray
) -> float:
    """Calculate autocorrelation difference"""
    differences = []
    max_lag = min(20, ori_data.shape[1] // 2)  # Limit lag for speed

    for feat in range(ori_data.shape[2]):
        ori_acf = []
        gen_acf = []

        for lag in range(1, max_lag + 1):
            # Calculate autocorrelation for original data
            ori_corr = np.corrcoef(
                ori_data[:, :-lag, feat].flatten(), ori_data[:, lag:, feat].flatten()
            )[0, 1]
            ori_acf.append(ori_corr if not np.isnan(ori_corr) else 0)

            # Calculate autocorrelation for generated data
            gen_corr = np.corrcoef(
                gen_data[:, :-lag, feat].flatten(), gen_data[:, lag:, feat].flatten()
            )[0, 1]
            gen_acf.append(gen_corr if not np.isnan(gen_corr) else 0)

        # Calculate difference
        diff = np.mean(np.abs(np.array(ori_acf) - np.array(gen_acf)))
        differences.append(diff)

    return np.mean(differences)


def calculate_skewness_difference(ori_data: np.ndarray, gen_data: np.ndarray) -> float:
    """Calculate skewness difference"""
    differences = []
    for feat in range(ori_data.shape[2]):
        ori_flat = ori_data[:, :, feat].flatten()
        gen_flat = gen_data[:, :, feat].flatten()

        ori_skew = calculate_skewness(ori_flat)
        gen_skew = calculate_skewness(gen_flat)

        differences.append(abs(ori_skew - gen_skew))

    return np.mean(differences)


def calculate_kurtosis_difference(ori_data: np.ndarray, gen_data: np.ndarray) -> float:
    """Calculate kurtosis difference"""
    differences = []
    for feat in range(ori_data.shape[2]):
        ori_flat = ori_data[:, :, feat].flatten()
        gen_flat = gen_data[:, :, feat].flatten()

        ori_kurt = calculate_kurtosis(ori_flat)
        gen_kurt = calculate_kurtosis(gen_flat)

        differences.append(abs(ori_kurt - gen_kurt))

    return np.mean(differences)


def calculate_skewness(data: np.ndarray) -> float:
    """Calculate skewness of data"""
    mean = np.mean(data)
    std = np.std(data)
    skew = np.mean(((data - mean) / std) ** 3)
    return skew if not np.isnan(skew) else 0


def calculate_kurtosis(data: np.ndarray) -> float:
    """Calculate kurtosis of data"""
    mean = np.mean(data)
    std = np.std(data)
    kurt = np.mean(((data - mean) / std) ** 4) - 3
    return kurt if not np.isnan(kurt) else 0


def calculate_discriminative_score(ori_data: np.ndarray, gen_data: np.ndarray) -> float:
    """Calculate discriminative score (simplified version)"""
    # Simple statistical measure to distinguish real vs generated
    ori_mean = np.mean(ori_data, axis=(0, 1))
    gen_mean = np.mean(gen_data, axis=(0, 1))

    ori_std = np.std(ori_data, axis=(0, 1))
    gen_std = np.std(gen_data, axis=(0, 1))

    # Calculate difference in distribution characteristics
    mean_diff = np.mean(np.abs(ori_mean - gen_mean))
    std_diff = np.mean(np.abs(ori_std - gen_std))

    # Normalize to [0, 1] range
    score = (mean_diff + std_diff) / 2
    return min(score, 1.0)


def calculate_predictive_score(ori_data: np.ndarray, gen_data: np.ndarray) -> float:
    """Calculate predictive score (simplified version)"""
    # Use generated data to predict original data patterns
    differences = []

    for feat in range(ori_data.shape[2]):
        # Simple prediction: use generated data mean to predict original
        gen_mean = np.mean(gen_data[:, :, feat])
        ori_flat = ori_data[:, :, feat].flatten()

        # Calculate MAE between original and predicted (constant mean)
        mae = np.mean(np.abs(ori_flat - gen_mean))
        differences.append(mae)

    return np.mean(differences)


def calculate_contextual_fid(ori_data: np.ndarray, gen_data: np.ndarray) -> float:
    """Calculate Contextual-FID (simplified version)"""
    # Calculate mean and covariance for each feature
    ori_means = np.mean(ori_data, axis=(0, 1))
    gen_means = np.mean(gen_data, axis=(0, 1))

    ori_covs = []
    gen_covs = []

    for feat in range(ori_data.shape[2]):
        ori_feat = ori_data[:, :, feat].flatten()
        gen_feat = gen_data[:, :, feat].flatten()

        ori_covs.append(np.cov(ori_feat))
        gen_covs.append(np.cov(gen_feat))

    # Calculate FID-like score
    mean_diff = np.sum((ori_means - gen_means) ** 2)

    cov_diff = 0
    for ori_cov, gen_cov in zip(ori_covs, gen_covs):
        try:
            covmean = sqrtm(ori_cov.dot(gen_cov))
            if np.iscomplexobj(covmean):
                covmean = covmean.real
            cov_diff += np.trace(ori_cov + gen_cov - 2.0 * covmean)
        except:
            cov_diff += np.trace(ori_means + gen_means)

    return abs(mean_diff + cov_diff)


def generate_data(model_results: Dict[str, Any], num_samples: int = None) -> np.ndarray:
    """Generate synthetic time series data using trained model"""
    if model_results.get("is_deep_model", False):
        # Deep learning models already have generated data
        return model_results["generated_data"]

    # Traditional ML models - use the existing generation logic
    model = model_results["model"]
    train_sequences = model_results["train_sequences"]
    model_type = model_results.get("model_type", "")

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

    # Generate synthetic sequences by sampling from training distribution
    # For simple models, we'll use the model to predict next values
    # and reconstruct sequences
    generated_sequences = []

    for _ in range(num_samples):
        # Sample a random training sequence as starting point
        start_idx = np.random.randint(0, len(train_sequences))
        start_sequence = train_sequences[start_idx].copy()

        # Generate next values using the model
        for i in range(len(start_sequence) - 1):
            # Use model to predict next value
            input_seq = start_sequence[i : i + 1].reshape(1, -1)
            next_val = model.predict(input_seq)[0]
            start_sequence[i + 1] = next_val

        generated_sequences.append(start_sequence)

    return np.array(generated_sequences)
