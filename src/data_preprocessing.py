import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple, Union
from statsmodels.tsa.stattools import acf
from scipy.signal import argrelextrema
import warnings

warnings.filterwarnings("ignore")


def data_preprocessing(
    data: Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]],
    target_col: str = None,
    sequence_length: int = None,
) -> Union[
    Tuple[np.ndarray, np.ndarray, MinMaxScaler],
    Tuple[np.ndarray, np.ndarray, np.ndarray, MinMaxScaler],
]:
    """Preprocess data with domain adaptation support for Air dataset"""

    # Handle domain adaptation setup for Air dataset
    if isinstance(data, tuple) and len(data) == 2:
        # Air dataset with source and target domains
        source_data, target_data = data
        return _preprocess_air_domains(source_data, target_data, sequence_length)

    # Handle single dataset (Google Stock, etc.)
    if isinstance(data, pd.DataFrame):
        ori_data = data.values
    else:
        ori_data = data

    # Create sequences
    sequences = _create_sequences_simple(ori_data, sequence_length)
    sequences = np.array(sequences, copy=True)  # Fix for read-only array

    # Shuffle and split
    np.random.shuffle(sequences)
    train_size = int(0.9 * len(sequences))
    train_sequences = sequences[:train_size]
    test_sequences = sequences[train_size:]

    # Normalize to [0,1]
    scaler = MinMaxScaler()
    train_sequences = scaler.fit_transform(
        train_sequences.reshape(-1, train_sequences.shape[-1])
    ).reshape(train_sequences.shape)
    test_sequences = scaler.transform(
        test_sequences.reshape(-1, test_sequences.shape[-1])
    ).reshape(test_sequences.shape)

    return train_sequences, test_sequences, scaler


def _preprocess_air_domains(
    source_data: pd.DataFrame, target_data: pd.DataFrame, sequence_length: int = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, MinMaxScaler]:
    """Preprocess Air dataset with domain adaptation"""

    # Determine sequence length from source data first
    if sequence_length is None:
        sequence_length = _find_length_simple(
            source_data.drop(["city", "domain"], axis=1).values[:, 0]
        )

    print(f"Using consistent sequence length: {sequence_length}")

    # Process source domain (Tianjin) for training
    source_sequences = _create_sequences_simple(
        source_data.drop(["city", "domain"], axis=1).values, sequence_length
    )
    source_sequences = np.array(source_sequences, copy=True)

    # Process target domains (Beijing, Guangzhou, Shenzhen) for evaluation
    # Use the SAME sequence length for consistency
    target_sequences = _create_sequences_simple(
        target_data.drop(["city", "domain"], axis=1).values, sequence_length
    )
    target_sequences = np.array(target_sequences, copy=True)

    # Shuffle source sequences for training
    np.random.shuffle(source_sequences)

    # Split source data into train/validation (90/10)
    train_size = int(0.9 * len(source_sequences))
    train_sequences = source_sequences[:train_size]
    val_sequences = source_sequences[train_size:]

    # Use target data as test set for domain adaptation evaluation
    test_sequences = target_sequences

    # Normalize all data using source domain statistics (domain adaptation principle)
    scaler = MinMaxScaler()
    train_sequences = scaler.fit_transform(
        train_sequences.reshape(-1, train_sequences.shape[-1])
    ).reshape(train_sequences.shape)

    val_sequences = scaler.transform(
        val_sequences.reshape(-1, val_sequences.shape[-1])
    ).reshape(val_sequences.shape)

    test_sequences = scaler.transform(
        test_sequences.reshape(-1, test_sequences.shape[-1])
    ).reshape(test_sequences.shape)

    print(f"Domain Adaptation Setup:")
    print(
        f"  Source Domain (Tianjin): Train={len(train_sequences)}, Val={len(val_sequences)}"
    )
    print(f"  Target Domains (BJ/GZ/SZ): Test={len(test_sequences)}")
    print(
        f"  Sequence shapes: Train={train_sequences.shape}, Test={test_sequences.shape}, Val={val_sequences.shape}"
    )

    return train_sequences, test_sequences, val_sequences, scaler


def _create_sequences_simple(
    data: np.ndarray, sequence_length: int = None
) -> np.ndarray:
    """Create overlapping sequences with stride 1"""
    if sequence_length is None:
        sequence_length = _find_length_simple(data[:, 0])

    if len(data) >= sequence_length:
        sequences = np.lib.stride_tricks.sliding_window_view(
            data, (sequence_length, data.shape[1])
        )
        return np.squeeze(sequences, axis=1)
    else:
        return np.array([data])


def _find_length_simple(data: np.ndarray) -> int:
    """Find optimal sequence length using autocorrelation"""
    data = data[: min(20000, len(data))]
    base = 3
    nobs = len(data)
    nlags = int(min(10 * np.log10(nobs), nobs - 1))

    try:
        auto_corr = acf(data, nlags=nlags, fft=True)[base:]
        local_max = argrelextrema(auto_corr, np.greater)[0]

        if len(local_max) > 0:
            max_local_max = np.argmax([auto_corr[lcm] for lcm in local_max])
            length = local_max[max_local_max] + base
            # Ensure consistent sequence length within reasonable bounds
            length = max(30, min(100, length))
            # Round to nearest multiple of 5 for consistency
            length = round(length / 5) * 5
            return length
    except:
        pass

    # Default to a stable, consistent length
    return 125
