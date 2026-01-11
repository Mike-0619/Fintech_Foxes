"""
Baseline model training: Logistic Regression, XGBoost, Local NN, Central NN

FIXED: Added ECE computation for all models.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from models import CreditNet
from config import NUM_CLIENTS, BATCH_SIZE
from calibration import get_ece


def train_logistic_regression(X_train, y_train, X_test, y_test, hyperparams=None):
    """Train Logistic Regression baseline."""
    print("\n[1/4] Training Logistic Regression...")
    print("-" * 50)

    model = LogisticRegression(
        class_weight='balanced',
        random_state=42,
        max_iter=hyperparams.get('max_iter', 1000) if hyperparams else 1000,
        C=hyperparams.get('C', 1.0) if hyperparams else 1.0
    )
    model.fit(X_train, y_train)

    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs > 0.5).astype(int)

    # FIXED: Compute ECE for sklearn models
    ece = get_ece(probs, y_test)

    metrics = {
        'accuracy': accuracy_score(y_test, preds),
        'auc': roc_auc_score(y_test, probs),
        'f1': f1_score(y_test, preds, zero_division=0),
        'ece': ece
    }

    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  AUC:      {metrics['auc']:.4f}")
    print(f"  F1:       {metrics['f1']:.4f}")
    print(f"  ECE:      {metrics['ece']:.4f}")

    return model, metrics


def train_xgboost(X_train, y_train, X_test, y_test, hyperparams=None):
    """Train XGBoost baseline."""
    print("\n[2/4] Training XGBoost...")
    print("-" * 50)

    try:
        from xgboost import XGBClassifier

        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

        model = XGBClassifier(
            n_estimators=hyperparams.get('n_estimators', 100) if hyperparams else 100,
            max_depth=hyperparams.get('max_depth', 6) if hyperparams else 6,
            learning_rate=hyperparams.get('learning_rate', 0.1) if hyperparams else 0.1,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss',
            verbosity=0
        )
        model.fit(X_train, y_train)

        probs = model.predict_proba(X_test)[:, 1]
        preds = (probs > 0.5).astype(int)
        
        # FIXED: Compute ECE for sklearn models
        ece = get_ece(probs, y_test)

        metrics = {
            'accuracy': accuracy_score(y_test, preds),
            'auc': roc_auc_score(y_test, probs),
            'f1': f1_score(y_test, preds, zero_division=0),
            'ece': ece
        }

        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  AUC:      {metrics['auc']:.4f}")
        print(f"  F1:       {metrics['f1']:.4f}")
        print(f"  ECE:      {metrics['ece']:.4f}")

        return model, metrics

    except ImportError:
        print("  ⚠ XGBoost not installed. Skipping...")
        return None, None


def train_local_neural_networks(X_train, y_train, X_test, y_test, client_indices,
                                 input_dim, device, hyperparams=None):
    """Train local neural networks (one per client)."""
    print("\n[3/4] Training Local Neural Networks...")
    print("-" * 50)

    client_metrics = []

    # Get hyperparams or use defaults
    hidden = hyperparams.get('hidden', [64, 32]) if hyperparams else [64, 32]
    dropout = hyperparams.get('dropout', 0.2) if hyperparams else 0.2
    lr = hyperparams.get('lr', 0.005) if hyperparams else 0.005

    for client_id in range(NUM_CLIENTS):
        idx = client_indices[client_id]
        X_client = X_train[idx]
        y_client = y_train[idx]
        
        # Skip empty clients
        if len(idx) == 0:
            print(f"  Client {client_id}: No data, skipping")
            continue

        # Train local model
        local_model = CreditNet(input_dim, hidden_layers=hidden, dropout=dropout).to(device)
        local_model.train()

        optimizer = torch.optim.Adam(local_model.parameters(), lr=lr)
        criterion = nn.BCELoss()

        ds = TensorDataset(
            torch.Tensor(X_client),
            torch.Tensor(y_client).float().unsqueeze(1)
        )
        # Adaptive batch size for small clients
        actual_batch_size = min(BATCH_SIZE, len(X_client) // 2) if len(X_client) > BATCH_SIZE else max(8, len(X_client) // 4)
        dl = DataLoader(ds, batch_size=actual_batch_size, shuffle=True, drop_last=False)

        # Train for 15 epochs
        for epoch in range(15):
            for bx, by in dl:
                bx, by = bx.to(device), by.to(device)
                optimizer.zero_grad()
                loss = criterion(local_model(bx), by)
                loss.backward()
                optimizer.step()

        # Evaluate
        local_model.eval()
        with torch.no_grad():
            preds = local_model(torch.Tensor(X_test).to(device))
            probs = preds.cpu().numpy().flatten()
            preds_binary = (probs > 0.5).astype(int)

        # FIXED: Compute ECE for local NNs
        ece = get_ece(probs, y_test)

        metrics = {
            'accuracy': accuracy_score(y_test, preds_binary),
            'auc': roc_auc_score(y_test, probs),
            'f1': f1_score(y_test, preds_binary, zero_division=0),
            'ece': ece
        }
        client_metrics.append(metrics)

        print(f"  Client {client_id}: Acc={metrics['accuracy']:.4f}, AUC={metrics['auc']:.4f}, ECE={metrics['ece']:.4f}")

    # Average metrics
    avg_metrics = {
        'accuracy': np.mean([m['accuracy'] for m in client_metrics]),
        'auc': np.mean([m['auc'] for m in client_metrics]),
        'f1': np.mean([m['f1'] for m in client_metrics]),
        'ece': np.mean([m['ece'] for m in client_metrics])
    }

    print(f"\n  Average: Acc={avg_metrics['accuracy']:.4f}, AUC={avg_metrics['auc']:.4f}, ECE={avg_metrics['ece']:.4f}")

    return avg_metrics


def train_central_neural_network(X_train, y_train, X_test, y_test, input_dim, device, hyperparams=None):
    """Train central neural network on all data."""
    print("\n[4/4] Training Central Neural Network...")
    print("-" * 50)

    # Get hyperparams or use defaults
    hidden = hyperparams.get('hidden', [64, 32]) if hyperparams else [64, 32]
    dropout = hyperparams.get('dropout', 0.2) if hyperparams else 0.2
    lr = hyperparams.get('lr', 0.005) if hyperparams else 0.005

    model = CreditNet(input_dim, hidden_layers=hidden, dropout=dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    ds = TensorDataset(
        torch.Tensor(X_train),
        torch.Tensor(y_train).float().unsqueeze(1)
    )
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)

    # Train for 15 epochs
    model.train()
    for epoch in range(15):
        for bx, by in dl:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            optimizer.step()

    # Evaluate
    model.eval()
    with torch.no_grad():
        preds = model(torch.Tensor(X_test).to(device))
        probs = preds.cpu().numpy().flatten()
        preds_binary = (probs > 0.5).astype(int)

    # FIXED: Compute ECE for central NN
    ece = get_ece(probs, y_test)

    metrics = {
        'accuracy': accuracy_score(y_test, preds_binary),
        'auc': roc_auc_score(y_test, probs),
        'f1': f1_score(y_test, preds_binary, zero_division=0),
        'ece': ece
    }

    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  AUC:      {metrics['auc']:.4f}")
    print(f"  F1:       {metrics['f1']:.4f}")
    print(f"  ECE:      {metrics['ece']:.4f}")

    return model, metrics


def train_all_baselines(X_train, y_train, X_test, y_test, client_indices, input_dim, device, tuned_params=None):
    """Train all baseline models.

    Args:
        tuned_params: Dict with model-specific hyperparameters, e.g.:
            {
                'Logistic Regression': {'max_iter': 1000, 'C': 1.0},
                'XGBoost': {'n_estimators': 100, 'max_depth': 6},
                'Central NN': {'hidden': [64, 32], 'dropout': 0.2, 'lr': 0.005},
                'Local NN': {'hidden': [64, 32], 'dropout': 0.2, 'lr': 0.005}
            }
    """
    print("\n" + "=" * 80)
    print("TRAINING BASELINE MODELS")
    print("=" * 80)

    results = {}

    # Get hyperparams for each model
    lr_params = tuned_params.get('LogisticRegression', {}).get('params', {}) if tuned_params else None
    xgb_params = tuned_params.get('XGBOOST', {}).get('params', {}) if tuned_params else None
    local_params = tuned_params.get('Local NN', {}).get('params', {}) if tuned_params else None
    central_params = tuned_params.get('Central NN', {}).get('params', {}) if tuned_params else None

    # Logistic Regression
    lr_model, lr_metrics = train_logistic_regression(X_train, y_train, X_test, y_test, lr_params)
    results['Logistic Regression'] = lr_metrics

    # XGBoost
    _, xgb_metrics = train_xgboost(X_train, y_train, X_test, y_test, xgb_params)
    if xgb_metrics:
        results['XGBoost'] = xgb_metrics

    # Local NN
    local_metrics = train_local_neural_networks(
        X_train, y_train, X_test, y_test, client_indices, input_dim, device, local_params
    )
    results['Local NN (Avg)'] = local_metrics

    # Central NN
    central_model, central_metrics = train_central_neural_network(
        X_train, y_train, X_test, y_test, input_dim, device, central_params
    )
    results['Central NN'] = central_metrics

    return results, central_model
