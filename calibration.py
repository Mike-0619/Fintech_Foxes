"""
Calibration methods: Platt Scaling, Temperature Scaling, Beta Calibration, FedCal
"""

import numpy as np
import torch

from models import PlattScaler, TemperatureScaler, FedCalPlattScaler, BetaCalibrator
from config import NUM_CLIENTS


def get_ece(probs, labels, n_bins=10):
    """
    Calculate Expected Calibration Error (ECE).

    Args:
        probs: Predicted probabilities
        labels: True labels
        n_bins: Number of bins for calibration

    Returns:
        ECE score (lower is better)
    """
    probs = probs.flatten()
    labels = labels.flatten()
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        mask = (probs > bin_boundaries[i]) & (probs <= bin_boundaries[i + 1])
        if np.any(mask):
            bin_weight = mask.sum() / len(probs)
            bin_accuracy = labels[mask].mean()
            bin_confidence = probs[mask].mean()
            ece += np.abs(bin_confidence - bin_accuracy) * bin_weight

    return ece


def platt_scaling(global_model, X_calib, y_calib, device):
    """
    Train a single global PlattScaler.

    Args:
        global_model: Trained FL model
        X_calib: Calibration features
        y_calib: Calibration labels
        device: PyTorch device

    Returns:
        Trained PlattScaler
    """
    print("\nPlatt Scaling")
    print("-" * 40)

    global_model.eval()
    calib_logits = global_model.main(torch.Tensor(X_calib).to(device)).detach()
    calib_labels = torch.Tensor(y_calib).float().unsqueeze(1).to(device)

    scaler = PlattScaler().to(device)
    scaler.fit(calib_logits, calib_labels)

    return scaler


def temperature_scaling(global_model, X_calib, y_calib, device):
    """
    Train a single global TemperatureScaler.

    Args:
        global_model: Trained FL model
        X_calib: Calibration features
        y_calib: Calibration labels
        device: PyTorch device

    Returns:
        Trained TemperatureScaler
    """
    print("\nTemperature Scaling")
    print("-" * 40)

    global_model.eval()
    calib_logits = global_model.main(torch.Tensor(X_calib).to(device)).detach()
    calib_labels = torch.Tensor(y_calib).float().unsqueeze(1).to(device)

    scaler = TemperatureScaler().to(device)
    scaler.fit(calib_logits, calib_labels)

    return scaler


def beta_scaling(global_model, X_calib, y_calib, device):
    """
    Train a Beta Calibrator (Kull et al., 2017).
    
    More flexible than Platt with 3 parameters (a, b, c).
    Handles asymmetric miscalibration better.

    Args:
        global_model: Trained FL model
        X_calib: Calibration features
        y_calib: Calibration labels
        device: PyTorch device

    Returns:
        Trained BetaCalibrator
    """
    print("\nBeta Calibration")
    print("-" * 40)

    global_model.eval()
    calib_logits = global_model.main(torch.Tensor(X_calib).to(device)).detach()
    calib_labels = torch.Tensor(y_calib).float().unsqueeze(1).to(device)

    scaler = BetaCalibrator().to(device)
    scaler.fit(calib_logits, calib_labels)

    return scaler


def fedcal_calibration(global_model, client_indices, X_calib, y_calib, X_train, y_train, device,
                      num_clients=NUM_CLIENTS):
    """
    FedCal: Client-Specific Calibration (Peng et al., 2024).
    
    FIXED: Now uses client-specific data distributions based on client_indices
    instead of random subsets of the global calibration set.

    Algorithm:
    1. Train one PlattScaler per client using calibration data that matches
       their training data distribution
    2. Aggregate parameters (A and B) via weighted averaging
    3. Create global scaler with aggregated parameters

    Args:
        global_model: Trained FL model
        client_indices: Dict mapping client_id -> train indices
        X_calib: Global calibration features
        y_calib: Global calibration labels
        X_train: Training features (for class distribution reference)
        y_train: Training labels (for class distribution reference)
        device: PyTorch device
        num_clients: Number of clients

    Returns:
        FedCalPlattScaler with aggregated parameters, dict of client scalers
    """
    print("\nFedCal Client-Specific Calibration")
    print("=" * 60)

    # Phase 1: Train client-specific scalers
    client_scalers = {}
    client_A_values = []
    client_B_values = []
    client_weights = []  # Weight by client size

    global_model.eval()

    for client_id in range(num_clients):
        # Get client's class distribution from training data
        client_train_idx = client_indices[client_id]
        client_y_train = y_train[client_train_idx]
        
        # Skip empty clients
        if len(client_train_idx) == 0:
            print(f"Client {client_id}: No data, skipping")
            continue
        
        # Calculate class distribution for this client
        client_class_ratio = client_y_train.mean() if len(client_y_train) > 0 else 0.5
        
        # Sample calibration data to match client's class distribution
        # This simulates each client having their own validation set
        np.random.seed(42 + client_id)
        
        # Get indices for each class in calibration set
        pos_calib_idx = np.where(y_calib == 1)[0]
        neg_calib_idx = np.where(y_calib == 0)[0]
        
        # Sample based on client's class distribution
        samples_per_client = min(200, len(X_calib) // num_clients)
        n_pos = int(samples_per_client * client_class_ratio)
        n_neg = samples_per_client - n_pos
        
        # Handle edge cases
        n_pos = min(n_pos, len(pos_calib_idx))
        n_neg = min(n_neg, len(neg_calib_idx))
        
        if n_pos == 0 and n_neg == 0:
            print(f"Client {client_id}: Not enough calibration samples, skipping")
            continue
        
        sampled_pos = np.random.choice(pos_calib_idx, size=n_pos, replace=False) if n_pos > 0 else np.array([])
        sampled_neg = np.random.choice(neg_calib_idx, size=n_neg, replace=False) if n_neg > 0 else np.array([])
        calib_indices = np.concatenate([sampled_pos, sampled_neg]).astype(int)
        
        X_client_calib = X_calib[calib_indices]
        y_client_calib = y_calib[calib_indices]

        # Get logits from global model
        client_logits = global_model.main(
            torch.Tensor(X_client_calib).to(device)
        ).detach()
        client_labels = torch.Tensor(y_client_calib).float().unsqueeze(1).to(device)

        # Train client-specific scaler
        client_scaler = FedCalPlattScaler().to(device)
        client_scaler.fit(client_logits, client_labels)

        # Store parameters
        client_scalers[client_id] = client_scaler
        client_A_values.append(client_scaler.A.detach().cpu())
        client_B_values.append(client_scaler.B.detach().cpu())
        client_weights.append(len(client_train_idx))  # Weight by training data size

        print(f"Client {client_id}: A={client_scaler.A.item():.4f}, B={client_scaler.B.item():.4f} (class_ratio={client_class_ratio:.3f})")

    # Phase 2: Weighted aggregation of parameters
    total_weight = sum(client_weights)
    weights_normalized = [w / total_weight for w in client_weights]
    
    A_weighted = sum(w * A for w, A in zip(weights_normalized, client_A_values))
    B_weighted = sum(w * B for w, B in zip(weights_normalized, client_B_values))

    print(f"\nWeighted Aggregated Parameters: A={A_weighted.item():.4f}, B={B_weighted.item():.4f}")
    print(f"A std dev: {torch.stack(client_A_values).std().item():.4f}")
    print(f"B std dev: {torch.stack(client_B_values).std().item():.4f}")

    # Phase 3: Create global scaler with aggregated parameters
    fedcal_scaler = FedCalPlattScaler().to(device)
    fedcal_scaler.A.data.fill_(A_weighted.item())
    fedcal_scaler.B.data.fill_(B_weighted.item())

    return fedcal_scaler, client_scalers


def compare_calibration_methods(global_model, X_calib, y_calib, X_eval, y_eval,
                                client_indices, X_train, y_train, device):
    """
    Compare three calibration methods: Platt, Temperature, FedCal.

    Args:
        global_model: Trained FL model
        X_calib: Calibration features
        y_calib: Calibration labels
        X_eval: Evaluation features
        y_eval: Evaluation labels
        client_indices: Dict mapping client_id -> train indices
        X_train: Training features (for FedCal class distribution)
        y_train: Training labels (for FedCal class distribution)
        device: PyTorch device

    Returns:
        Dictionary of scalers and metrics
    """
    from sklearn.metrics import accuracy_score, f1_score

    print("\n" + "=" * 80)
    print("FOUR-WAY CALIBRATION COMPARISON")
    print("=" * 80)

    # Train all four scalers
    platt_scaler = platt_scaling(global_model, X_calib, y_calib, device)
    temp_scaler = temperature_scaling(global_model, X_calib, y_calib, device)
    beta_scaler = beta_scaling(global_model, X_calib, y_calib, device)
    fedcal_scaler, _ = fedcal_calibration(global_model, client_indices, X_calib, y_calib, X_train, y_train, device)

    # Evaluate on eval set
    eval_logits = global_model.main(torch.Tensor(X_eval).to(device)).detach()
    eval_y = torch.Tensor(y_eval).float().unsqueeze(1).to(device)
    eval_labels_np = eval_y.cpu().numpy()

    # Get predictions from all methods
    raw_probs = torch.sigmoid(eval_logits).cpu().numpy()
    platt_probs = platt_scaler(eval_logits).detach().cpu().numpy()
    temp_probs = temp_scaler(eval_logits).detach().cpu().numpy()
    beta_probs = beta_scaler(eval_logits).detach().cpu().numpy()
    fedcal_probs = fedcal_scaler(eval_logits).detach().cpu().numpy()

    methods = {
        'Uncalibrated': raw_probs,
        'Platt Scaling': platt_probs,
        'Temperature Scaling': temp_probs,
        'Beta Calibration': beta_probs,
        'FedCal': fedcal_probs
    }

    results = {}
    for method_name, probs in methods.items():
        preds = (probs > 0.5).astype(int)
        acc = accuracy_score(eval_labels_np, preds)
        f1 = f1_score(eval_labels_np, preds, zero_division=0)
        ece = get_ece(probs, eval_labels_np)
        results[method_name] = {'Accuracy': acc, 'F1': f1, 'ECE': ece}

    # Print results
    print("\n" + "-" * 80)
    print("Calibration Methods Comparison:")
    print("-" * 80)
    for method, metrics in results.items():
        print(f"{method:20s}: Acc={metrics['Accuracy']:.4f}, F1={metrics['F1']:.4f}, ECE={metrics['ECE']:.4f}")

    return {
        'scalers': {
            'platt': platt_scaler,
            'temp': temp_scaler,
            'beta': beta_scaler,
            'fedcal': fedcal_scaler
        },
        'results': results
    }
