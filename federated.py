"""
Federated Learning training functions
"""

import copy
import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict

from config import LEARNING_RATE, LOCAL_EPOCHS, BATCH_SIZE
from models import CreditNet, CreditLSTM


def get_device(model):
    """Get the device of a model safely."""
    return next(model.parameters()).device


def train_client_mlp(model, X, y, epochs=LOCAL_EPOCHS, lr=LEARNING_RATE):
    """
    Train MLP model on local client data with class-specific weighting.

    Args:
        model: CreditNet instance
        X: Client features [N, D]
        y: Client labels [N]
        epochs: Number of local epochs
        lr: Learning rate (default from config)

    Returns:
        state_dict: Model weights
    """
    from torch.utils.data import DataLoader, TensorDataset

    device = get_device(model)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Calculate client-specific class weights for imbalance handling
    pos_count = np.sum(y == 1)
    neg_count = np.sum(y == 0)
    if pos_count > 0:
        pos_weight = neg_count / pos_count
        # Clip extreme weights to prevent overfitting
        pos_weight = np.clip(pos_weight, 0.5, 10.0)
    else:
        pos_weight = 1.0
    
    # Use BCEWithLogitsLoss with pos_weight for numerical stability and class weighting
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))

    ds = TensorDataset(torch.Tensor(X), torch.Tensor(y).float().unsqueeze(1))
    # Use smaller batch size for potentially small client datasets
    actual_batch_size = min(BATCH_SIZE, len(X) // 2) if len(X) > BATCH_SIZE else max(8, len(X) // 4)
    dl = DataLoader(ds, batch_size=actual_batch_size, shuffle=True, drop_last=False)

    for _ in range(epochs):
        for bx, by in dl:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            # Use model.main() to get logits for BCEWithLogitsLoss
            logits = model.main(bx)
            loss = criterion(logits, by)
            loss.backward()
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

    return model.state_dict()


def train_client_lstm(model, X, y, epochs=LOCAL_EPOCHS, lr=LEARNING_RATE):
    """
    Train LSTM model on local client data with class-specific weighting.

    Args:
        model: CreditLSTM instance
        X: Client features [N, D]
        y: Client labels [N]
        epochs: Number of local epochs
        lr: Learning rate (default from config)

    Returns:
        state_dict: Model weights
    """
    from torch.utils.data import DataLoader, TensorDataset

    device = get_device(model)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Calculate client-specific class weights for imbalance handling
    pos_count = np.sum(y == 1)
    neg_count = np.sum(y == 0)
    if pos_count > 0:
        pos_weight = neg_count / pos_count
        # Clip extreme weights to prevent overfitting
        pos_weight = np.clip(pos_weight, 0.5, 10.0)
    else:
        pos_weight = 1.0
    
    # Use BCEWithLogitsLoss with pos_weight for numerical stability and class weighting
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))

    ds = TensorDataset(torch.Tensor(X), torch.Tensor(y).float().unsqueeze(1))
    # Use smaller batch size for potentially small client datasets
    actual_batch_size = min(BATCH_SIZE, len(X) // 2) if len(X) > BATCH_SIZE else max(8, len(X) // 4)
    dl = DataLoader(ds, batch_size=actual_batch_size, shuffle=True, drop_last=False)

    for _ in range(epochs):
        for bx, by in dl:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            # Use model.main() to get logits for BCEWithLogitsLoss
            logits = model.main(bx)
            loss = criterion(logits, by)
            loss.backward()
            # Gradient clipping for LSTM stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

    return model.state_dict()


def federated_averaging(weights_list, client_sizes=None):
    """
    FedAvg aggregation: Weighted average of weights across clients.
    
    FIXED: Now supports weighted averaging by client dataset size.

    Args:
        weights_list: List of state_dicts from clients
        client_sizes: Optional list of client dataset sizes for weighted averaging.
                     If None, uses simple (unweighted) averaging.

    Returns:
        Averaged state_dict
    """
    if client_sizes is None:
        # Simple averaging (original behavior)
        avg_weights = OrderedDict()
        for k in weights_list[0].keys():
            current_dtype = weights_list[0][k].dtype
            avg_weights[k] = torch.stack([w[k].float() for w in weights_list]).mean(dim=0).to(current_dtype)
        return avg_weights
    
    # Weighted averaging by client size
    total_samples = sum(client_sizes)
    weights = [size / total_samples for size in client_sizes]
    
    avg_weights = OrderedDict()
    for k in weights_list[0].keys():
        current_dtype = weights_list[0][k].dtype
        weighted_sum = sum(w * state_dict[k].float() for w, state_dict in zip(weights, weights_list))
        avg_weights[k] = weighted_sum.to(current_dtype)
    
    return avg_weights


def federated_training(model_class, model_kwargs, train_fn, X_train, y_train,
                      client_indices, X_test, y_test, device, num_rounds,
                      model_name="FL", hyperparams=None, use_weighted_avg=True):
    """
    Generic federated training loop.

    Args:
        model_class: Model class (CreditNet or CreditLSTM)
        model_kwargs: Arguments to pass to model constructor
        train_fn: Training function (train_client_mlp or train_client_lstm)
        X_train: Training features
        y_train: Training labels
        client_indices: Dict mapping client_id -> indices
        X_test: Test features
        y_test: Test labels
        device: PyTorch device
        num_rounds: Number of FL rounds
        model_name: Name for logging
        hyperparams: Dict with model-specific hyperparameters (e.g., {'lr': 0.01})
        use_weighted_avg: If True, weight client updates by dataset size (default True)

    Returns:
        Trained model, list of test accuracies per round
    """
    from config import NUM_CLIENTS

    print(f"\n{'='*60}")
    print(f"Federated Training: {model_name}")
    print(f"{'='*60}")

    # Initialize global model
    global_model = model_class(**model_kwargs).to(device)
    test_accs = []

    # Get learning rate from hyperparams or use default
    lr = hyperparams.get('lr', LEARNING_RATE) if hyperparams else LEARNING_RATE

    # Convert test data to tensor
    test_tensor_x = torch.Tensor(X_test).to(device)
    test_tensor_y = torch.Tensor(y_test).float().unsqueeze(1).to(device)

    # Get client sizes for weighted averaging
    client_sizes = [len(client_indices[i]) for i in range(NUM_CLIENTS)]
    
    print(f"Training for {num_rounds} rounds with {NUM_CLIENTS} clients...")
    print(f"Client sizes: {client_sizes}")
    print(f"Using {'weighted' if use_weighted_avg else 'unweighted'} FedAvg")
    if hyperparams:
        print(f"Using hyperparams: {hyperparams}")

    for round_num in range(1, num_rounds + 1):
        local_weights_list = []

        # Local training on each client
        for i in range(NUM_CLIENTS):
            idx = client_indices[i]
            
            # Skip clients with no data (shouldn't happen but safety check)
            if len(idx) == 0:
                print(f"  Warning: Client {i} has no data, skipping")
                continue
                
            local_weights = train_fn(
                copy.deepcopy(global_model),
                X_train[idx],
                y_train[idx],
                lr=lr
            )
            local_weights_list.append(local_weights)

        # FedAvg aggregation (weighted by client size)
        if use_weighted_avg:
            # Only include sizes for clients that were actually trained
            active_sizes = [s for s, w in zip(client_sizes, local_weights_list) if w is not None]
            global_weights = federated_averaging(local_weights_list, client_sizes=active_sizes)
        else:
            global_weights = federated_averaging(local_weights_list)
            
        global_model.load_state_dict(global_weights)

        # Evaluate
        global_model.eval()
        with torch.no_grad():
            preds = global_model(test_tensor_x)
            acc = ((preds > 0.5) == test_tensor_y).float().mean().item()
            test_accs.append(acc)

        print(f"  Round {round_num:2d}: Accuracy = {acc:.4f}")

    print(f"\n{model_name} Final Accuracy: {test_accs[-1]:.4f}")

    return global_model, test_accs
