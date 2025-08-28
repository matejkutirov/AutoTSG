#!/usr/bin/env python3
"""AutoML implementation using FLAML's search strategy for custom time series models"""

import flaml
from flaml import AutoML
from flaml.automl.model import BaseEstimator
from flaml import tune
import numpy as np
import time
from typing import Dict, Any, Tuple
from src.data_loading import data_loading
from src.data_preprocessing import data_preprocessing
from src.model_generation import model_generation, generate_data, save_generated_data
from src.evaluation import evaluation


class ARIMAEstimator(BaseEstimator):
    """FLAML wrapper for ARIMA model"""

    def __init__(self, **config):
        super().__init__(**config)
        self.config = config
        self._model = None
        self.train_sequences = None
        self.test_sequences = None

    @classmethod
    def search_space(cls, data_size, task):
        # data_size is a tuple (samples, features), we want the number of samples
        num_samples = data_size[0] if isinstance(data_size, tuple) else data_size
        return {
            "model_type": {"domain": tune.choice(["arima"]), "init_value": "arima"},
            "sample_size": {
                "domain": tune.randint(
                    100, min(5000, num_samples)
                ),  # Much larger range for budget scaling
                "init_value": 500,
            },
            "p": {
                "domain": tune.randint(0, 5),  # AR order
                "init_value": 1,
            },
            "d": {
                "domain": tune.randint(0, 2),  # Differencing order
                "init_value": 1,
            },
            "q": {
                "domain": tune.randint(0, 5),  # MA order
                "init_value": 1,
            },
        }

    def fit(self, X_train, y_train=None, **kwargs):
        # Store sequences and time budget for later use
        self.train_sequences = kwargs.get("train_sequences")
        self.test_sequences = kwargs.get("test_sequences")
        time_budget = kwargs.get("time_budget", None)

        try:
            # Fixed training: use all samples, no scaling
            n_samples = len(self.train_sequences)

            # Get individual budget for ARIMA (2.5% of total)
            individual_budget = kwargs.get("individual_budgets", {}).get(
                "arima", time_budget * 0.025 if time_budget else None
            )

            sample_train = self.train_sequences[:n_samples]
            sample_test = self.test_sequences[:n_samples]

            print(f"Training ARIMA with {len(sample_train)} samples")
            print(f"Time budget: {individual_budget}s")

            # Start timer for actual time enforcement
            model_start_time = time.time()

            # Train normally - ARIMA is fast
            model_results = model_generation(sample_train, sample_test, "arima")
            self._model = model_results["model"]

            actual_time = time.time() - model_start_time
            print(f"ARIMA training completed in {actual_time:.1f}s!")
            return self
        except Exception as e:
            raise RuntimeError(f"Failed to train ARIMA: {e}")

    def predict(self, X, **kwargs):
        if self._model is None:
            raise RuntimeError("Model not trained yet")

        # Create model results structure for generation
        model_results = {
            "model": self._model,
            "model_type": "arima",
            "train_sequences": self.train_sequences,
            "test_sequences": self.test_sequences,
            "is_deep_model": False,
        }

        # Generate synthetic data - ensure we generate exactly X.shape[0] samples
        num_samples = X.shape[0]

        # Check if generated_data exists, if not generate it on-the-fly
        if (
            "generated_data" in model_results
            and model_results["generated_data"] is not None
        ):
            generated_data = model_results["generated_data"]
        else:
            # Generate data on-the-fly using the trained model
            if hasattr(self._model, "get_prior_samples"):  # TimeVAE
                generated_data = self._model.get_prior_samples(num_samples)
            elif hasattr(self._model, "generate"):  # TimeGAN
                generated_data = self._model.generate(num_samples)
                generated_data = np.array(generated_data)
            elif hasattr(self._model, "sample"):  # FourierFlow, TimeTransformer
                generated_data = self._model.sample(num_samples)
                if hasattr(generated_data, "detach"):  # PyTorch tensor
                    generated_data = generated_data.detach().cpu().numpy()
            else:
                # Fallback to the original generate_data function
                generated_data = generate_data(model_results, num_samples=num_samples)

        # For ARIMA, we need to return a single value per sample
        # Use the mean across the sequence and features
        if generated_data.ndim == 3:  # (samples, seq_len, features)
            # Take the mean across sequence length and features
            single_values = np.mean(generated_data, axis=(1, 2))
        else:
            single_values = (
                np.mean(generated_data, axis=1)
                if generated_data.ndim == 2
                else generated_data.flatten()
            )

        # Ensure we return exactly num_samples values
        return single_values[:num_samples].reshape(-1, 1)


class ExponentialSmoothingEstimator(BaseEstimator):
    """FLAML wrapper for Exponential Smoothing model"""

    def __init__(self, **config):
        super().__init__(**config)
        self.config = config
        self._model = None
        self.train_sequences = None
        self.test_sequences = None

    @classmethod
    def search_space(cls, data_size, task):
        # data_size is a tuple (samples, features), we want the number of samples
        num_samples = data_size[0] if isinstance(data_size, tuple) else data_size
        return {
            "model_type": {
                "domain": tune.choice(["exp_smooth"]),
                "init_value": "exp_smooth",
            },
            "sample_size": {
                "domain": tune.randint(100, min(1000, num_samples)),
                "init_value": 500,
            },
        }

    def fit(self, X_train, y_train=None, **kwargs):
        # Store sequences and time budget for later use
        self.train_sequences = kwargs.get("train_sequences")
        self.test_sequences = kwargs.get("test_sequences")
        time_budget = kwargs.get("time_budget", None)

        try:
            # Use budget-aware sampling - scale sample size based on individual model budget
            base_samples = self.config.get("sample_size", len(self.train_sequences))

            # Get individual budget for Exponential Smoothing (10% of total)
            individual_budget = kwargs.get("individual_budgets", {}).get(
                "exp_smooth", time_budget * 0.1 if time_budget else None
            )

            if individual_budget is not None:
                # Estimate time per sample and scale accordingly
                # Exponential Smoothing is similar to ARIMA in speed
                estimated_time_per_sample = 0.001  # seconds per sample
                max_samples = int(individual_budget / estimated_time_per_sample)
                n_samples = min(base_samples, max_samples, len(self.train_sequences))
                print(
                    f"Exp_Smooth budget: {individual_budget}s, scaling samples from {base_samples} to {n_samples}"
                )
            else:
                n_samples = base_samples

            sample_train = self.train_sequences[:n_samples]
            sample_test = self.test_sequences[:n_samples]

            print(f"Training Exponential Smoothing with {len(sample_train)} samples...")
            model_results = model_generation(sample_train, sample_test, "exp_smooth")
            self._model = model_results["model"]
            print(f"Exponential Smoothing training completed!")
            return self
        except Exception as e:
            raise RuntimeError(f"Failed to train Exponential Smoothing: {e}")

    def predict(self, X, **kwargs):
        if self._model is None:
            raise RuntimeError("Model not trained yet")

        # Create model results structure for generation
        model_results = {
            "model": self._model,
            "model_type": "exp_smooth",
            "train_sequences": self.train_sequences,
            "test_sequences": self.test_sequences,
            "is_deep_model": False,
        }

        # Generate synthetic data - ensure we generate exactly X.shape[0] samples
        num_samples = X.shape[0]

        # Check if generated_data exists, if not generate it on-the-fly
        if (
            "generated_data" in model_results
            and model_results["generated_data"] is not None
        ):
            generated_data = model_results["generated_data"]
        else:
            # Generate data on-the-fly using the trained model
            if hasattr(self._model, "get_prior_samples"):  # TimeVAE
                generated_data = self._model.get_prior_samples(num_samples)
            elif hasattr(self._model, "generate"):  # TimeGAN
                generated_data = self._model.generate(num_samples)
                generated_data = np.array(generated_data)
            elif hasattr(self._model, "sample"):  # FourierFlow, TimeTransformer
                generated_data = self._model.sample(num_samples)
                if hasattr(generated_data, "detach"):  # PyTorch tensor
                    generated_data = generated_data.detach().cpu().numpy()
            else:
                # Fallback to the original generate_data function
                generated_data = generate_data(model_results, num_samples=num_samples)

        # For Exponential Smoothing, we need to return a single value per sample
        # Use the mean across the sequence and features
        if generated_data.ndim == 3:  # (samples, seq_len, features)
            # Take the mean across sequence length and features
            single_values = np.mean(generated_data, axis=(1, 2))
        else:
            single_values = (
                np.mean(generated_data, axis=1)
                if generated_data.ndim == 2
                else generated_data.flatten()
            )

        # Ensure we return exactly num_samples values
        return single_values[:num_samples].reshape(-1, 1)


class TimeVAEEstimator(BaseEstimator):
    """FLAML wrapper for TimeVAE model"""

    def __init__(self, **config):
        super().__init__(**config)
        self.config = config
        self._model = None
        self.train_sequences = None
        self.test_sequences = None

    @classmethod
    def search_space(cls, data_size, task):
        # data_size is a tuple (samples, features), we want the number of samples
        num_samples = data_size[0] if isinstance(data_size, tuple) else data_size
        return {
            "model_type": {"domain": tune.choice(["timevae"]), "init_value": "timevae"},
            "train_epochs": {
                "domain": tune.randint(1, 10),
                "init_value": 2,
            },  # cheap first
            "sample_size": {
                "domain": tune.randint(100, min(2000, num_samples)),
                "init_value": 1000,
            },
        }

    def fit(self, X_train, y_train=None, **kwargs):
        # Store sequences and time budget for later use
        self.train_sequences = kwargs.get("train_sequences")
        self.test_sequences = kwargs.get("test_sequences")
        time_budget = kwargs.get("time_budget", None)

        try:
            # Fixed training epochs: 100 for all deep models
            n_epochs = 100
            n_samples = len(self.train_sequences)

            # Get individual budget for TimeVAE (20% of total)
            individual_budget = kwargs.get("individual_budgets", {}).get(
                "timevae", time_budget * 0.2 if time_budget else None
            )

            sample_train = self.train_sequences[:n_samples]
            sample_test = self.test_sequences[:n_samples]

            print(f"Training TimeVAE...")
            print(f"Training for {n_epochs} epochs with {n_samples} samples")
            print(f"Time budget: {individual_budget}s")

            # Start timer for actual time enforcement
            model_start_time = time.time()

            # Train epoch by epoch with time budget checking
            for epoch in range(n_epochs):
                # Check if time budget exceeded
                if (
                    individual_budget is not None
                    and (time.time() - model_start_time) > individual_budget
                ):
                    print(
                        f"⏹ Stopping early: reached time budget ({individual_budget}s) at epoch {epoch}"
                    )
                    break

                # Train just one epoch at a time
                model_results = model_generation(
                    sample_train, sample_test, "timevae", max_epochs=1
                )
                self._model = model_results["model"]

            actual_time = time.time() - model_start_time
            print(f"TimeVAE training completed in {actual_time:.1f}s!")

            return self
        except Exception as e:
            raise RuntimeError(f"Failed to train TimeVAE: {e}")

    def predict(self, X, **kwargs):
        if self._model is None:
            raise RuntimeError("Model not trained yet")

        # Create model results structure for generation
        model_results = {
            "model": self._model,
            "model_type": "timevae",
            "train_sequences": self.train_sequences,
            "test_sequences": self.test_sequences,
            "is_deep_model": True,
        }

        # Generate synthetic data - ensure we generate exactly X.shape[0] samples
        num_samples = X.shape[0]

        # Check if generated_data exists, if not generate it on-the-fly
        if (
            "generated_data" in model_results
            and model_results["generated_data"] is not None
        ):
            generated_data = model_results["generated_data"]
        else:
            # Generate data on-the-fly using the trained model
            if hasattr(self._model, "get_prior_samples"):  # TimeVAE
                generated_data = self._model.get_prior_samples(num_samples)
            elif hasattr(self._model, "generate"):  # TimeGAN
                generated_data = self._model.generate(num_samples)
                generated_data = np.array(generated_data)
            elif hasattr(self._model, "sample"):  # FourierFlow, TimeTransformer
                generated_data = self._model.sample(num_samples)
                if hasattr(generated_data, "detach"):  # PyTorch tensor
                    generated_data = generated_data.detach().cpu().numpy()
            else:
                # Fallback to the original generate_data function
                generated_data = generate_data(model_results, num_samples=num_samples)

        # For TimeVAE, we need to return a single value per sample
        # Use the mean across the sequence and features
        if generated_data.ndim == 3:  # (samples, seq_len, features)
            # Take the mean across sequence length and features
            single_values = np.mean(generated_data, axis=(1, 2))
        else:
            single_values = (
                np.mean(generated_data, axis=1)
                if generated_data.ndim == 2
                else generated_data.flatten()
            )

        # Ensure we return exactly num_samples values
        return single_values[:num_samples].reshape(-1, 1)


class TimeGANEstimator(BaseEstimator):
    """FLAML wrapper for TimeGAN model"""

    def __init__(self, **config):
        super().__init__(**config)
        self.config = config
        self._model = None
        self.train_sequences = None
        self.test_sequences = None

    @classmethod
    def search_space(cls, data_size, task):
        # data_size is a tuple (samples, features), we want the number of samples
        num_samples = data_size[0] if isinstance(data_size, tuple) else data_size
        return {
            "model_type": {"domain": tune.choice(["timegan"]), "init_value": "timegan"},
            "train_epochs": {
                "domain": tune.randint(1, 5),
                "init_value": 1,
            },  # very cheap first
            "sample_size": {
                "domain": tune.randint(100, min(2000, num_samples)),
                "init_value": 1000,
            },
        }

    def fit(self, X_train, y_train=None, **kwargs):
        # Store sequences and time budget for later use
        self.train_sequences = kwargs.get("train_sequences")
        self.test_sequences = kwargs.get("test_sequences")
        time_budget = kwargs.get("time_budget", None)

        try:
            # Fixed training iterations: 100 for all deep models
            n_iterations = 100
            n_samples = len(self.train_sequences)

            # Get individual budget for TimeGAN (20% of total)
            individual_budget = kwargs.get("individual_budgets", {}).get(
                "timegan", time_budget * 0.2 if time_budget else None
            )

            sample_train = self.train_sequences[:n_samples]
            sample_test = self.test_sequences[:n_samples]

            print(f"Training TimeGAN...")
            print(f"Training for {n_iterations} iterations with {n_samples} samples")
            print(f"Time budget: {individual_budget}s")

            # Start timer for actual time enforcement
            model_start_time = time.time()

            # Train iteration by iteration with time budget checking
            for iteration in range(n_iterations):
                # Check if time budget exceeded
                if (
                    individual_budget is not None
                    and (time.time() - model_start_time) > individual_budget
                ):
                    print(
                        f"⏹ Stopping early: reached time budget ({individual_budget}s) at iteration {iteration}"
                    )
                    break

                # Train just one iteration at a time
                model_results = model_generation(
                    sample_train, sample_test, "timegan", max_epochs=1
                )
                self._model = model_results["model"]

            actual_time = time.time() - model_start_time
            print(f"TimeGAN training completed in {actual_time:.1f}s!")

            return self
        except Exception as e:
            raise RuntimeError(f"Failed to train TimeGAN: {e}")

    def predict(self, X, **kwargs):
        if self._model is None:
            raise RuntimeError("Model not trained yet")

        # Create model results structure for generation
        model_results = {
            "model": self._model,
            "model_type": "timegan",
            "train_sequences": self.train_sequences,
            "test_sequences": self.test_sequences,
            "is_deep_model": True,
        }

        # Generate synthetic data - ensure we generate exactly X.shape[0] samples
        num_samples = X.shape[0]

        # Check if generated_data exists, if not generate it on-the-fly
        if (
            "generated_data" in model_results
            and model_results["generated_data"] is not None
        ):
            generated_data = model_results["generated_data"]
        else:
            # Generate data on-the-fly using the trained model
            if hasattr(self._model, "get_prior_samples"):  # TimeVAE
                generated_data = self._model.get_prior_samples(num_samples)
            elif hasattr(self._model, "generate"):  # TimeGAN
                generated_data = self._model.generate(num_samples)
                generated_data = np.array(generated_data)
            elif hasattr(self._model, "sample"):  # FourierFlow, TimeTransformer
                generated_data = self._model.sample(num_samples)
                if hasattr(generated_data, "detach"):  # PyTorch tensor
                    generated_data = generated_data.detach().cpu().numpy()
            else:
                # Fallback to the original generate_data function
                generated_data = generate_data(model_results, num_samples=num_samples)

        # For TimeGAN, we need to return a single value per sample
        # Use the mean across the sequence and features
        if generated_data.ndim == 3:  # (samples, seq_len, features)
            # Take the mean across sequence length and features
            single_values = np.mean(generated_data, axis=(1, 2))
        else:
            single_values = (
                np.mean(generated_data, axis=1)
                if generated_data.ndim == 2
                else generated_data.flatten()
            )

        # Ensure we return exactly num_samples values
        return single_values[:num_samples].reshape(-1, 1)


class FourierFlowEstimator(BaseEstimator):
    """FLAML wrapper for FourierFlow model"""

    def __init__(self, **config):
        super().__init__(**config)
        self.config = config
        self._model = None
        self.train_sequences = None
        self.test_sequences = None

    @classmethod
    def search_space(cls, data_size, task):
        # data_size is a tuple (samples, features), we want the number of samples
        num_samples = data_size[0] if isinstance(data_size, tuple) else data_size
        return {
            "model_type": {
                "domain": tune.choice(["fourierflow"]),
                "init_value": "fourierflow",
            },
            "train_epochs": {"domain": tune.randint(1, 8), "init_value": 2},
            "sample_size": {
                "domain": tune.randint(100, min(2000, num_samples)),
                "init_value": 1000,
            },
        }

    def fit(self, X_train, y_train=None, **kwargs):
        # Store sequences for later use
        self.train_sequences = kwargs.get("train_sequences")
        self.test_sequences = kwargs.get("test_sequences")

        try:
            # Fixed training epochs: 100 for all deep models
            n_epochs = 100
            n_samples = len(self.train_sequences)

            # Get individual budget for FourierFlow (20% of total)
            time_budget = kwargs.get("time_budget", None)
            individual_budget = kwargs.get("individual_budgets", {}).get(
                "fourierflow", time_budget * 0.2 if time_budget else None
            )

            sample_train = self.train_sequences[:n_samples]
            sample_test = self.test_sequences[:n_samples]

            print(f"Training FourierFlow...")
            print(f"Training for {n_epochs} epochs with {n_samples} samples")
            print(f"Time budget: {individual_budget}s")

            # Start timer for actual time enforcement
            model_start_time = time.time()

            # Train epoch by epoch with time budget checking
            for epoch in range(n_epochs):
                # Check if time budget exceeded
                if (
                    individual_budget is not None
                    and (time.time() - model_start_time) > individual_budget
                ):
                    print(
                        f"⏹ Stopping early: reached time budget ({individual_budget}s) at epoch {epoch}"
                    )
                    break

                # Train just one epoch at a time
                model_results = model_generation(
                    sample_train, sample_test, "fourierflow", max_epochs=1
                )
                self._model = model_results["model"]

            actual_time = time.time() - model_start_time
            print(f"FourierFlow training completed in {actual_time:.1f}s!")

            return self
        except Exception as e:
            raise RuntimeError(f"Failed to train FourierFlow: {e}")

    def predict(self, X, **kwargs):
        if self._model is None:
            raise RuntimeError("Model not trained yet")

        # Create model results structure for generation
        model_results = {
            "model": self._model,
            "model_type": "fourierflow",
            "train_sequences": self.train_sequences,
            "test_sequences": self.test_sequences,
            "is_deep_model": True,
        }

        # Generate synthetic data - ensure we generate exactly X.shape[0] samples
        num_samples = X.shape[0]

        # Check if generated_data exists, if not generate it on-the-fly
        if (
            "generated_data" in model_results
            and model_results["generated_data"] is not None
        ):
            generated_data = model_results["generated_data"]
        else:
            # Generate data on-the-fly using the trained model
            if hasattr(self._model, "get_prior_samples"):  # TimeVAE
                generated_data = self._model.get_prior_samples(num_samples)
            elif hasattr(self._model, "generate"):  # TimeGAN
                generated_data = self._model.generate(num_samples)
                generated_data = np.array(generated_data)
            elif hasattr(self._model, "sample"):  # FourierFlow, TimeTransformer
                generated_data = self._model.sample(num_samples)
                if hasattr(generated_data, "detach"):  # PyTorch tensor
                    generated_data = generated_data.detach().cpu().numpy()
            else:
                # Fallback to the original generate_data function
                generated_data = generate_data(model_results, num_samples=num_samples)

        # For FourierFlow, we need to return a single value per sample
        # Use the mean across the sequence and features
        if generated_data.ndim == 3:  # (samples, seq_len, features)
            # Take the mean across sequence length and features
            single_values = np.mean(generated_data, axis=(1, 2))
        else:
            single_values = (
                np.mean(generated_data, axis=1)
                if generated_data.ndim == 2
                else generated_data.flatten()
            )

        # Ensure we return exactly num_samples values
        return single_values[:num_samples].reshape(-1, 1)


class TimeTransformerEstimator(BaseEstimator):
    """FLAML wrapper for TimeTransformer model"""

    def __init__(self, **config):
        super().__init__(**config)
        self.config = config
        self._model = None
        self.train_sequences = None
        self.test_sequences = None

    @classmethod
    def search_space(cls, data_size, task):
        # data_size is a tuple (samples, features), we want the number of samples
        num_samples = data_size[0] if isinstance(data_size, tuple) else data_size
        return {
            "model_type": {
                "domain": tune.choice(["timetransformer"]),
                "init_value": "timetransformer",
            },
            "train_epochs": {"domain": tune.randint(1, 6), "init_value": 2},
            "sample_size": {
                "domain": tune.randint(100, min(2000, num_samples)),
                "init_value": 1000,
            },
        }

    def fit(self, X_train, y_train=None, **kwargs):
        # Store sequences for later use
        self.train_sequences = kwargs.get("train_sequences")
        self.test_sequences = kwargs.get("test_sequences")

        try:
            # Fixed training epochs: 100 for all deep models
            n_epochs = 100
            n_samples = len(self.train_sequences)

            # Get individual budget for TimeTransformer (35% of total)
            time_budget = kwargs.get("time_budget", None)
            individual_budget = kwargs.get("individual_budgets", {}).get(
                "timetransformer", time_budget * 0.35 if time_budget else None
            )

            sample_train = self.train_sequences[:n_samples]
            sample_test = self.test_sequences[:n_samples]

            print(f"Training TimeTransformer...")
            print(f"Training for {n_epochs} epochs with {n_samples} samples")
            print(f"Time budget: {individual_budget}s")

            # Start timer for actual time enforcement
            model_start_time = time.time()

            # Train epoch by epoch with time budget checking
            for epoch in range(n_epochs):
                # Check if time budget exceeded
                if (
                    individual_budget is not None
                    and (time.time() - model_start_time) > individual_budget
                ):
                    print(
                        f"⏹ Stopping early: reached time budget ({individual_budget}s) at epoch {epoch}"
                    )
                    break

                # Train just one epoch at a time
                model_results = model_generation(
                    sample_train, sample_test, "timetransformer", max_epochs=1
                )
                self._model = model_results["model"]

            actual_time = time.time() - model_start_time
            print(f"TimeTransformer training completed in {actual_time:.1f}s!")

            self._model = model_results["model"]
            return self
        except Exception as e:
            raise RuntimeError(f"Failed to train TimeTransformer: {e}")

    def predict(self, X, **kwargs):
        if self._model is None:
            raise RuntimeError("Model not trained yet")

        # Create model results structure for generation
        model_results = {
            "model": self._model,
            "model_type": "timetransformer",
            "train_sequences": self.train_sequences,
            "test_sequences": self.test_sequences,
            "is_deep_model": True,
        }

        # Generate synthetic data - ensure we generate exactly X.shape[0] samples
        num_samples = X.shape[0]

        # Check if generated_data exists, if not generate it on-the-fly
        if (
            "generated_data" in model_results
            and model_results["generated_data"] is not None
        ):
            generated_data = model_results["generated_data"]
        else:
            # Generate data on-the-fly using the trained model
            if hasattr(self._model, "get_prior_samples"):  # TimeVAE
                generated_data = self._model.get_prior_samples(num_samples)
            elif hasattr(self._model, "generate"):  # TimeGAN
                generated_data = self._model.generate(num_samples)
                generated_data = np.array(generated_data)
            elif hasattr(self._model, "sample"):  # FourierFlow, TimeTransformer
                generated_data = self._model.sample(num_samples)
                if hasattr(generated_data, "detach"):  # PyTorch tensor
                    generated_data = generated_data.detach().cpu().numpy()
            else:
                # Fallback to the original generate_data function
                generated_data = generate_data(model_results, num_samples=num_samples)

        # For TimeTransformer, we need to return a single value per sample
        # Use the mean across the sequence and features
        if generated_data.ndim == 3:  # (samples, seq_len, features)
            # Take the mean across sequence length and features
            single_values = np.mean(generated_data, axis=(1, 2))
        else:
            single_values = (
                np.mean(generated_data, axis=1)
                if generated_data.ndim == 2
                else generated_data.flatten()
            )

        # Ensure we return exactly num_samples values
        return single_values[:num_samples].reshape(-1, 1)


# NOTE: You need to modify model_generation.py to accept max_epochs parameter
# Example modification for deep models:
# def model_generation(train_sequences, test_sequences, model_type, max_epochs=None):
#     if model_type in ["timevae", "timegan", "fourierflow", "timetransformer"]:
#         # Use max_epochs if provided, otherwise use default
#         epochs = max_epochs if max_epochs is not None else 100  # default
#         # Pass epochs to model training
#     # ... rest of the function


def automl_model_selection(
    dataset_name: str,
    time_budget: int = 300,  # 5 minutes default
    metric: str = "RMSE",
    models_to_test: list = None,
) -> Dict[str, Any]:
    """Automatically select the best time series forecasting model using FLAML's search strategy"""

    print(f"Starting AutoML model selection for {dataset_name}")
    print(f"Time budget: {time_budget} seconds")
    print(f"Optimization metric: {metric}")
    print("=" * 50)

    # Load and preprocess data
    data = data_loading(dataset_name)
    train_sequences, test_sequences, scaler = data_preprocessing(data)

    # Define models to test
    if models_to_test is None or (
        len(models_to_test) == 1 and models_to_test[0].lower() == "all"
    ):
        models_to_test = [
            "arima",
            "exp_smooth",
            "timevae",
            "timegan",
            "fourierflow",
            "timetransformer",
        ]

    print(f"Testing models: {', '.join(models_to_test)}")
    print("=" * 50)

    # Prepare data for FLAML (flatten sequences)
    X_train = train_sequences.reshape(train_sequences.shape[0], -1)
    y_train = train_sequences[:, -1, 0]  # Use last time step of first feature as target

    # Initialize AutoML
    automl = AutoML()

    # Register all estimators
    estimator_mapping = {
        "arima": ARIMAEstimator,
        "exp_smooth": ExponentialSmoothingEstimator,
        "timevae": TimeVAEEstimator,
        "timegan": TimeGANEstimator,
        "fourierflow": FourierFlowEstimator,
        "timetransformer": TimeTransformerEstimator,
    }

    # Only register the models we want to test
    for model_name in models_to_test:
        if model_name in estimator_mapping:
            automl.add_learner(model_name, estimator_mapping[model_name])

    # Calculate time budget distribution per model
    # ARIMA: 10%, Exp_Smooth: 10%, Deep models: 20% each
    time_distribution = {
        "arima": 0.025,
        "exp_smooth": 0.025,
        "timevae": 0.20,
        "timegan": 0.20,
        "fourierflow": 0.20,
        "timetransformer": 0.35,
    }

    # Calculate individual time budgets
    individual_budgets = {}
    for model_name in models_to_test:
        if model_name in time_distribution:
            individual_budgets[model_name] = int(
                time_budget * time_distribution[model_name]
            )
        else:
            individual_budgets[model_name] = int(
                time_budget / len(models_to_test)
            )  # Equal distribution for unknown models

    print(f"\nTime Budget Distribution:")
    for model_name, budget in individual_budgets.items():
        print(f"  {model_name}: {budget}s ({100 * budget / time_budget:.1f}%)")
    print()

    # Define AutoML settings
    automl_settings = {
        "time_budget": time_budget,
        "metric": "rmse",  # Use our custom metric
        "task": "regression",  # FLAML requires a task
        "estimator_list": models_to_test,  # Use the models we registered
        "log_file_name": f"automl_{dataset_name}.log",
        "verbose": 2,
        "n_jobs": 1,  # Sequential execution for time series models
        "early_stop": False,
        "max_iter": 1000,  # Increase max iterations
        "eval_method": "holdout",
        "individual_budgets": individual_budgets,  # Pass individual budgets to estimators
    }

    # Run Sequential Training with Time Budget Enforcement
    print("Running Sequential Training with Time Budget Enforcement...")
    print(f"Time budget: {time_budget} seconds")
    print("=" * 50)

    # We'll train models sequentially, each within their allocated time
    start_time = time.time()

    # Store results for each model
    all_results = []
    best_score = float("inf")
    best_model = None
    best_config = None

    # Train each model sequentially within its time budget
    for model_name in models_to_test:
        if model_name not in individual_budgets:
            continue

        model_budget = individual_budgets[model_name]
        print(f"\n{'=' * 20} Training {model_name.upper()} {'=' * 20}")
        print(f"Allocated time: {model_budget}s")

        # Create estimator instance
        estimator_class = estimator_mapping[model_name]
        estimator = estimator_class()

        # Train the model with time enforcement
        try:
            model_start_time = time.time()

            # Train the model
            estimator.fit(
                X_train=X_train,
                y_train=y_train,
                train_sequences=train_sequences,
                test_sequences=test_sequences,
                time_budget=time_budget,
                individual_budgets=individual_budgets,
            )

            # Evaluate the model
            y_pred = estimator.predict(
                X_train[:100]
            )  # Predict on subset for evaluation
            model_score = np.sqrt(np.mean((y_train[:100] - y_pred.flatten()) ** 2))

            model_time = time.time() - model_start_time
            print(
                f"{model_name} training completed in {model_time:.1f}s (budget: {model_budget}s)"
            )
            print(f"{model_name} RMSE score: {model_score:.4f}")

            # Store results
            result = {
                "model_name": model_name,
                "model": estimator._model,
                "score": model_score,
                "time_used": model_time,
                "budget": model_budget,
                "estimator": estimator,
            }
            all_results.append(result)

            # Update best model if this one is better
            if model_score < best_score:
                best_score = model_score
                best_model = estimator._model
                best_config = {"model_type": model_name, "score": model_score}

        except Exception as e:
            print(f"Error training {model_name}: {e}")
            continue

    actual_time = time.time() - start_time
    print(f"\nSequential training completed in {actual_time:.1f} seconds")
    print(
        f"Time budget used: {actual_time:.1f}/{time_budget} seconds ({100 * actual_time / time_budget:.1f}%)"
    )

    # Check if we have any results
    if not all_results:
        raise RuntimeError("No models were successfully trained")

    print(f"\nBest model: {best_config.get('model_type', 'unknown')}")
    print(f"Best config: {best_config}")
    print(f"Best score: {best_score:.4f}")

    # Create model results structure for reference
    model_results = {
        "model": best_model,
        "model_type": best_config.get("model_type", "unknown"),
        "train_sequences": train_sequences,
        "test_sequences": test_sequences,
        "is_deep_model": best_config.get("model_type")
        in ["timevae", "timegan", "fourierflow", "timetransformer"],
        "training_time": time_budget,
        "automl_results": {
            "best_config": best_config,
            "best_score": best_score,
        },
    }

    print(f"\nAutoML Model Selection Complete!")
    print(f"Dataset: {dataset_name}")
    print(f"Time budget: {time_budget} seconds")
    print(f"Total models tested: {len(models_to_test)}")
    print(f"Best model: {best_config.get('model_type', 'unknown')}")
    print(f"Best score (RMSE): {best_score:.4f}")
    print(f"Best configuration: {best_config}")

    print(f"\nSequential Training Summary:")
    for result in all_results:
        print(
            f"✓ {result['model_name']}: {result['time_used']:.1f}s used, {result['budget']}s budget, RMSE: {result['score']:.4f}"
        )

    print(
        f"\nWinner: {best_config.get('model_type', 'unknown')} with RMSE: {best_score:.4f}"
    )
    print(f"Reason: Best performance among all sequentially trained models")

    return {
        "best_model": best_model,
        "best_model_type": best_config.get("model_type"),
        "best_config": best_config,
        "best_score": best_score,
        "automl_results": model_results["automl_results"],
    }


def quick_automl_test(
    dataset_name: str = "google", time_budget: int = 120, models_to_test: list = None
):
    """Quick test of AutoML functionality"""

    print(f"Quick AutoML test for {dataset_name} (time budget: {time_budget}s)")
    print("=" * 50)

    try:
        results = automl_model_selection(
            dataset_name, time_budget, models_to_test=models_to_test
        )

        print(f"\nAutoML completed successfully!")
        print(f"Best model: {results['best_model_type']}")
        print(f"Best score: {results['best_score']:.4f}")
        print(f"Generated data shape: {results['generated_data'].shape}")

        return results

    except Exception as e:
        print(f"AutoML failed: {e}")
        import traceback

        traceback.print_exc()
        return None


if __name__ == "__main__":
    # Quick test with Google stock data
    quick_automl_test("google", 120)
