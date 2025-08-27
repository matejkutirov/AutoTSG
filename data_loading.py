import pandas as pd
import numpy as np
import yfinance as yf
from ucimlrepo import fetch_ucirepo
from typing import Union, Optional, Tuple


def data_loading(
    dataset_name: str, file_path: Optional[str] = None
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.DataFrame]]:
    """Load datasets with domain adaptation support for Air dataset"""
    if dataset_name == "google":
        return yf.download("GOOGL", start="2004-01-01", end="2024-01-01")

    elif dataset_name == "air":
        # Domain adaptation setup as per the paper:
        # Tianjin (TJ) as source domain
        # Beijing (BJ), Guangzhou (GZ), Shenzhen (SZ) as target domains

        # Load source domain (Tianjin)
        source_data = pd.read_csv("data/Air/Tianjin_train.csv", index_col=0)
        source_data["city"] = "Tianjin"
        source_data["domain"] = "source"

        # Load target domains
        target_cities = ["Beijing", "Guangzhou", "Shenzhen"]
        target_data_frames = []

        for city in target_cities:
            train_data = pd.read_csv(f"data/Air/{city}_train.csv", index_col=0)
            train_data["city"] = city
            train_data["domain"] = "target"
            target_data_frames.append(train_data)

        target_data = pd.concat(target_data_frames, ignore_index=True)

        # Clean data: keep only numerical features
        source_data = _clean_air_data(source_data)
        target_data = _clean_air_data(target_data)

        # Return both source and target data for domain adaptation
        return source_data, target_data

    elif dataset_name == "appliances":
        # Load Appliances Energy Prediction dataset from UCI
        print("Loading Appliances Energy Prediction dataset...")
        appliances_energy_prediction = fetch_ucirepo(id=374)

        # Get features and targets
        X = appliances_energy_prediction.data.features

        # Drop non-continuous columns: date, lights
        X = X.drop(columns=["date", "lights"])

        # Clean the data
        data = _clean_uci_data(X)

        print(f"Loaded {data.shape[0]} samples with {data.shape[1]} features")
        return data

    elif dataset_name == "eeg":
        # Load EEG Eye State dataset from UCI
        print("Loading EEG Eye State dataset...")
        eeg_eye_state = fetch_ucirepo(id=264)

        # Get only the EEG features (exclude eyeDetection target)
        X = eeg_eye_state.data.features

        # Clean the features data
        data = _clean_uci_data(X)

        print(f"Loaded {data.shape[0]} EEG samples with {data.shape[1]} features")
        return data

    else:
        raise ValueError(
            f"Dataset {dataset_name} not supported. Available datasets: google, air, appliances, eeg"
        )


def _clean_air_data(data: pd.DataFrame) -> pd.DataFrame:
    """Clean Air dataset by keeping only numerical features"""

    # The data already has numerical values, just ensure proper types
    # Keep only the numerical features and metadata columns

    # Define the numerical feature columns (excluding city and domain)
    numerical_features = [
        "weather",
        "temperature",
        "pressure",
        "humidity",
        "wind_speed",
        "wind_direction",
        "PM10_Concentration",
        "NO2_Concentration",
        "CO_Concentration",
        "O3_Concentration",
        "SO2_Concentration",
        "PM25_Concentration",
    ]

    # Keep only numerical features and metadata
    metadata_cols = [col for col in data.columns if col in ["city", "domain"]]
    keep_cols = numerical_features + metadata_cols

    # Filter to only keep existing columns
    existing_cols = [col for col in keep_cols if col in data.columns]
    data = data[existing_cols]

    # Ensure all numerical columns are float
    for col in data.columns:
        if col not in ["city", "domain"]:
            data[col] = pd.to_numeric(data[col], errors="coerce").fillna(0.0)

    # Handle any remaining missing values
    data = data.fillna(method="ffill").fillna(method="bfill").fillna(0.0)

    return data


def _clean_uci_data(data: pd.DataFrame) -> pd.DataFrame:
    """Clean UCI datasets by handling missing values and ensuring proper data types"""

    # Handle missing values
    data = data.fillna(method="ffill").fillna(method="bfill").fillna(0.0)

    # Ensure all columns are numeric
    for col in data.columns:
        if data[col].dtype == "object":
            # Try to convert to numeric, if fails, drop the column
            try:
                data[col] = pd.to_numeric(data[col], errors="coerce")
                data[col] = data[col].fillna(0.0)
            except:
                print(f"Warning: Dropping non-numeric column {col}")
                data = data.drop(columns=[col])

    # Remove any rows with infinite values
    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.fillna(0.0)

    return data


def get_air_domains():
    """Get source and target domain information for Air dataset"""
    return {
        "source_domain": "Tianjin",
        "target_domains": ["Beijing", "Guangzhou", "Shenzhen"],
        "description": "Tianjin as source domain, other cities as target domains for domain adaptation",
    }
