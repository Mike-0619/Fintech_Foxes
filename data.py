"""
Data loading and preprocessing module
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
import ssl
import os
import joblib

from config import (
    DATASET_ID, TEST_SIZE, VAL_SIZE, CALIB_SIZE, RANDOM_STATE,
    NUM_CLIENTS, DIRICHLET_ALPHA, UCI_FEATURE_NAMES, BATCH_SIZE
)

# SSL Fix for data fetching
# Note: This is a workaround for SSL certificate issues on some systems.
# In production, use proper certificate configuration instead.
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context


def load_data(output_dir=None):
    """
    Load and preprocess the Default of Credit Card Clients dataset.
    
    IMPORTANT: This function returns RAW data. Use create_splits() which will
    fit the scaler on training data only to avoid data leakage.

    Args:
        output_dir: Optional directory to save the preprocessor for inference

    Returns:
        X: Raw feature matrix (unscaled)
        y: Labels
        feature_names: List of feature names
    """
    print("Loading data...")
    dataset = fetch_openml(data_id=DATASET_ID, as_frame=True, parser='auto')
    df = dataset.frame
    y = dataset.target.astype(int).values
    X = df.drop(columns=dataset.target_names)

    # Apply UCI feature names
    if len(X.columns) == len(UCI_FEATURE_NAMES):
        X.columns = UCI_FEATURE_NAMES
    
    feature_names = X.columns.tolist()

    print(f"  Loaded: {X.shape[0]} samples, {X.shape[1]} features")
    return X.values, y, feature_names


def create_splits(X, y, output_dir=None):
    """
    Create train/val/test splits with proper preprocessing.
    
    FIXED: Scaler is now fit on TRAINING data only to prevent data leakage.
    The scaler is saved to output_dir for inference.

    Args:
        X: Raw feature matrix
        y: Labels
        output_dir: Optional directory to save the scaler

    Returns:
        X_train, X_calib, X_eval, X_test, y_train, y_calib, y_eval, y_test
    """
    print("Creating train/val/test split...")

    # Initial split: 80% dev, 20% test
    X_dev, X_test, y_dev, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # Development split: 70% train, 30% val
    X_train, X_val, y_train, y_val = train_test_split(
        X_dev, y_dev, test_size=VAL_SIZE, random_state=RANDOM_STATE, stratify=y_dev
    )

    # Split validation into calibration and evaluation
    X_calib, X_eval, y_calib, y_eval = train_test_split(
        X_val, y_val, test_size=CALIB_SIZE, random_state=RANDOM_STATE, stratify=y_val
    )

    # FIX: Fit scaler on TRAINING data only to prevent data leakage
    print("  Fitting StandardScaler on training data only...")
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_calib = scaler.transform(X_calib)
    X_eval = scaler.transform(X_eval)
    X_test = scaler.transform(X_test)
    
    # Save scaler for inference
    if output_dir is not None:
        scaler_path = os.path.join(output_dir, 'scaler.joblib')
        joblib.dump(scaler, scaler_path)
        print(f"  [SAVED] Scaler to {scaler_path}")

    print(f"  Train:   {len(X_train)} samples")
    print(f"  Calib:   {len(X_calib)} samples")
    print(f"  Eval:    {len(X_eval)} samples")
    print(f"  Test:    {len(X_test)} samples")

    return X_train, X_calib, X_eval, X_test, y_train, y_calib, y_eval, y_test


def dirichlet_partition(y, num_clients=NUM_CLIENTS, alpha=DIRICHLET_ALPHA, seed=RANDOM_STATE):
    """
    Partition data across clients using Dirichlet distribution for Non-IID data.

    Args:
        y: Label array
        num_clients: Number of clients
        alpha: Dirichlet concentration (lower = more skew)
        seed: Random seed

    Returns:
        Dictionary mapping client_id -> numpy array of sample indices
    """
    np.random.seed(seed)
    N = len(y)
    indices = {i: [] for i in range(num_clients)}

    classes = np.unique(y)
    for c in classes:
        idx_k = np.where(y == c)[0]
        np.random.shuffle(idx_k)

        # Sample proportions from Dirichlet
        proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
        proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
        idx_split = np.split(idx_k, proportions)

        for i in range(num_clients):
            indices[i].extend(idx_split[i].tolist())

    # Convert to numpy arrays for consistent indexing
    indices = {i: np.array(idx) for i, idx in indices.items()}

    # Print heterogeneity stats
    print("\nClient Label Distributions:")
    ratios = []
    for i in range(num_clients):
        y_local = y[indices[i]]
        dist = np.bincount(y_local, minlength=2)
        ratio = dist[1] / (dist[0] + dist[1]) if len(dist) > 1 else 0
        ratios.append(ratio)
        print(f"  Client {i}: {len(indices[i])} samples, class dist {dist} (Ratio: {ratio:.3f})")

    ratio_std = np.std(ratios)
    print(f"\nHeterogeneity Metric (Std Dev): {ratio_std:.4f}")

    return indices


def create_dataloader(X, y, batch_size=BATCH_SIZE, shuffle=True, drop_last=False):
    """
    Create a PyTorch DataLoader.
    
    FIXED: drop_last=False by default to avoid losing data with small batches.
    """
    ds = TensorDataset(torch.Tensor(X), torch.Tensor(y).float().unsqueeze(1))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)


def get_tensors(X, y, device):
    """Convert numpy arrays to PyTorch tensors."""
    return torch.Tensor(X).to(device), torch.Tensor(y).float().unsqueeze(1).to(device)


def load_scaler(scaler_path):
    """Load a saved scaler for inference."""
    return joblib.load(scaler_path)
