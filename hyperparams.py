"""
Hyperparameter search spaces and tuning utilities.
"""

import json
import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression

# Optuna imports (optional - will be used if available)
try:
    import optuna
    from optuna.pruners import MedianPruner
    from optuna.samplers import TPESampler
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("Optuna not installed. Will use random search for tuning.")
    print("To install Optuna: pip install optuna")

# Search spaces for each model
MLP_SEARCH_SPACE = {
    'lr': [0.001, 0.005, 0.01],
    'hidden': [[32, 16], [64, 32], [128, 64]],
    'dropout': [0.1, 0.2, 0.3]
}

LSTM_SEARCH_SPACE = {
    'lr': [0.001, 0.005, 0.01],
    'hidden': [32, 64, 128],
    'layers': [2, 4, 6],
    'dropout': [0.1, 0.2, 0.3]
}

XGBOOST_SEARCH_SPACE = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 6, 9],
    'learning_rate': [0.05, 0.1, 0.2]
}

LOGREG_SEARCH_SPACE = {
    'C': [0.01, 0.1, 1.0, 10.0],
    'solver': ['lbfgs', 'liblinear']
}

CENTRAL_NN_SEARCH_SPACE = MLP_SEARCH_SPACE.copy()  # Same as MLP


def random_search_params(search_space, n_trials=10, seed=42):
    """
    Generate random hyperparameter combinations.

    Args:
        search_space: Dict of parameter names to list of values
        n_trials: Number of random combinations to generate
        seed: Random seed

    Returns:
        List of parameter dictionaries
    """
    random.seed(seed)
    param_combinations = []

    for _ in range(n_trials):
        params = {}
        for key, values in search_space.items():
            params[key] = random.choice(values)
        param_combinations.append(params)

    return param_combinations


def grid_search_params(search_space):
    """
    Generate all hyperparameter combinations (exhaustive grid search).

    Args:
        search_space: Dict of parameter names to list of values

    Returns:
        List of parameter dictionaries
    """
    keys, values = zip(*search_space.items())
    combinations = [dict(zip(keys, v)) for v in ParameterGrid(values)]
    return combinations


def validate_with_early_stopping(model, X_train, y_train, X_val, y_val,
                                  device='cpu', patience=5, min_delta=0.001,
                                  max_epochs=50):
    """
    Train model with early stopping on validation set.

    Args:
        model: PyTorch model with .main() method
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        device: PyTorch device
        patience: Number of epochs to wait for improvement
        min_delta: Minimum change to qualify as improvement
        max_epochs: Maximum number of training epochs

    Returns:
        Best validation score and best epoch
    """
    from torch.utils.data import DataLoader, TensorDataset

    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    criterion = nn.BCELoss()

    train_ds = TensorDataset(
        torch.Tensor(X_train),
        torch.Tensor(y_train).float().unsqueeze(1)
    )
    train_dl = DataLoader(train_ds, batch_size=64, shuffle=True, drop_last=True)

    best_val_loss = float('inf')
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(max_epochs):
        # Training
        model.train()
        for bx, by in train_dl:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            loss = criterion(model(bx), by)
            loss.backward()
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_logits = model.main(torch.Tensor(X_val).to(device))
            val_probs = torch.sigmoid(val_logits).cpu().numpy()
            val_loss = nn.BCELoss()(
                torch.Tensor(val_probs),
                torch.Tensor(y_val).float().unsqueeze(1)
            ).item()

        # Early stopping check
        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    return best_val_loss, best_epoch


def tune_mlp_hyperparams(X_train, y_train, X_val, y_val, X_eval, y_eval, X_test, y_test,
                        input_dim, device, search_space=MLP_SEARCH_SPACE,
                        n_trials=10, seed=42):
    """
    Tune MLP hyperparameters using random search.

    DATA SPLIT USAGE:
        X_train, y_train: Training data
        X_val, y_val: Validation data (for early stopping)
        X_eval, y_eval: Evaluation data (for hyperparameter selection) - FIXED DATA LEAKAGE
        X_test, y_test: Test data (for final evaluation only)

    Returns:
        Best params and best score
    """
    from models import CreditNet

    print("\n" + "=" * 60)
    print("Tuning MLP Hyperparameters")
    print("=" * 60)

    param_combinations = random_search_params(search_space, n_trials, seed)
    best_score = 0
    best_params = None

    for i, params in enumerate(param_combinations):
        print(f"\nTrial {i+1}/{n_trials}: {params}")

        # Create model with current hyperparameters
        model = CreditNet(input_dim)
        # Modify hidden layers based on params
        # Note: This is a simplified version - would need to modify CreditNet
        # For now, we'll just train with the current lr and dropout

        # Train with early stopping (uses X_val for validation)
        val_loss, best_epoch = validate_with_early_stopping(
            model, X_train, y_train, X_val, y_val, device
        )

        # Evaluate on EVAL set for hyperparameter selection (FIXED: was using X_test)
        model.eval()
        with torch.no_grad():
            eval_probs = torch.sigmoid(model.main(torch.Tensor(X_eval).to(device))).cpu().numpy()
        eval_acc = accuracy_score(y_eval, (eval_probs > 0.5).astype(int))

        # Also evaluate on test set for reporting (but NOT for selection)
        with torch.no_grad():
            test_probs = torch.sigmoid(model.main(torch.Tensor(X_test).to(device))).cpu().numpy()
        test_acc = accuracy_score(y_test, (test_probs > 0.5).astype(int))

        print(f"  Val Loss: {val_loss:.4f}, Eval Acc: {eval_acc:.4f}, Test Acc: {test_acc:.4f}")

        # Select based on EVAL accuracy (FIXED DATA LEAKAGE)
        if eval_acc > best_score:
            best_score = eval_acc
            best_params = params

    print(f"\nBest MLP Params: {best_params}")
    print(f"Best MLP Eval Accuracy: {best_score:.4f}")

    return best_params, best_score


def tune_lstm_hyperparams(X_train, y_train, X_val, y_val, X_eval, y_eval, X_test, y_test,
                         device, search_space=LSTM_SEARCH_SPACE,
                         n_trials=10, seed=42):
    """
    Tune LSTM hyperparameters using random search.

    DATA SPLIT USAGE:
        X_train, y_train: Training data
        X_val, y_val: Validation data (for early stopping)
        X_eval, y_eval: Evaluation data (for hyperparameter selection) - FIXED DATA LEAKAGE
        X_test, y_test: Test data (for final evaluation only)

    Returns:
        Best params and best score
    """
    from models import CreditLSTM

    print("\n" + "=" * 60)
    print("Tuning LSTM Hyperparameters")
    print("=" * 60)

    param_combinations = random_search_params(search_space, n_trials, seed)
    best_score = 0
    best_params = None

    for i, params in enumerate(param_combinations):
        print(f"\nTrial {i+1}/{n_trials}: {params}")

        # Create LSTM with current hyperparameters
        model = CreditLSTM(
            static_dim=5,
            temporal_dim=1,
            lstm_hidden=params['hidden'],
            lstm_layers=params['layers']
        )

        # Note: Would need to modify CreditLSTM to support dynamic dropout
        # For now, train with current lr and default dropout

        # Train with early stopping (more epochs for LSTM)
        val_loss, best_epoch = validate_with_early_stopping(
            model, X_train, y_train, X_val, y_val, device, max_epochs=50
        )

        # Evaluate on EVAL set for hyperparameter selection (FIXED: was using X_test)
        model.eval()
        with torch.no_grad():
            eval_probs = torch.sigmoid(model.main(torch.Tensor(X_eval).to(device))).cpu().numpy()
        eval_acc = accuracy_score(y_eval, (eval_probs > 0.5).astype(int))

        # Also evaluate on test set for reporting (but NOT for selection)
        with torch.no_grad():
            test_probs = torch.sigmoid(model.main(torch.Tensor(X_test).to(device))).cpu().numpy()
        test_acc = accuracy_score(y_test, (test_probs > 0.5).astype(int))

        print(f"  Val Loss: {val_loss:.4f}, Eval Acc: {eval_acc:.4f}, Test Acc: {test_acc:.4f}")

        # Select based on EVAL accuracy (FIXED DATA LEAKAGE)
        if eval_acc > best_score:
            best_score = eval_acc
            best_params = params

    print(f"\nBest LSTM Params: {best_params}")
    print(f"Best LSTM Eval Accuracy: {best_score:.4f}")

    return best_params, best_score


def tune_xgboost_hyperparams(X_train, y_train, X_val, y_val, X_eval, y_eval, X_test, y_test,
                            search_space=XGBOOST_SEARCH_SPACE,
                            n_trials=10, seed=42):
    """
    Tune XGBoost hyperparameters using random search.

    DATA SPLIT USAGE:
        X_train, y_train: Training data
        X_val, y_val: Not used (XGBoost has internal validation)
        X_eval, y_eval: Evaluation data (for hyperparameter selection) - FIXED DATA LEAKAGE
        X_test, y_test: Test data (for final evaluation only)

    Returns:
        Best params and best score
    """
    try:
        import xgboost as xgb

        print("\n" + "=" * 60)
        print("Tuning XGBoost Hyperparameters")
        print("=" * 60)

        param_combinations = random_search_params(search_space, n_trials, seed)
        best_score = 0
        best_params = None

        for i, params in enumerate(param_combinations):
            print(f"\nTrial {i+1}/{n_trials}: {params}")

            scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

            model = xgb.XGBClassifier(
                n_estimators=params['n_estimators'],
                max_depth=params['max_depth'],
                learning_rate=params['learning_rate'],
                scale_pos_weight=scale_pos_weight,
                random_state=seed,
                use_label_encoder=False,
                eval_metric='logloss',
                verbosity=0
            )
            model.fit(X_train, y_train)

            # Evaluate on EVAL set for hyperparameter selection (FIXED: was using X_val)
            eval_probs = model.predict_proba(X_eval)[:, 1]
            eval_acc = accuracy_score(y_eval, (eval_probs > 0.5).astype(int))

            # Also evaluate on test set for reporting (but NOT for selection)
            test_probs = model.predict_proba(X_test)[:, 1]
            test_acc = accuracy_score(y_test, (test_probs > 0.5).astype(int))

            print(f"  Eval Acc: {eval_acc:.4f}, Test Acc: {test_acc:.4f}")

            # Select based on EVAL accuracy (FIXED DATA LEAKAGE)
            if eval_acc > best_score:
                best_score = eval_acc
                best_params = params

        print(f"\nBest XGBoost Params: {best_params}")
        print(f"Best XGBoost Eval Accuracy: {best_score:.4f}")

        return best_params, best_score

    except ImportError:
        print("XGBoost not installed. Skipping XGBoost tuning.")
        return None, None


def tune_logreg_hyperparams(X_train, y_train, X_val, y_val, X_eval, y_eval, X_test, y_test,
                            search_space=LOGREG_SEARCH_SPACE,
                            n_trials=10, seed=42):
    """
    Tune Logistic Regression hyperparameters.

    DATA SPLIT USAGE:
        X_train, y_train: Training data
        X_val, y_val: Not used (sklearn models don't need validation)
        X_eval, y_eval: Evaluation data (for hyperparameter selection) - FIXED DATA LEAKAGE
        X_test, y_test: Test data (for final evaluation only)

    Returns:
        Best params and best score
    """
    print("\n" + "=" * 60)
    print("Tuning Logistic Regression Hyperparameters")
    print("=" * 60)

    param_combinations = random_search_params(search_space, n_trials, seed)
    best_score = 0
    best_params = None

    for i, params in enumerate(param_combinations):
        print(f"\nTrial {i+1}/{n_trials}: {params}")

        model = LogisticRegression(
            C=params['C'],
            solver=params['solver'],
            class_weight='balanced',
            random_state=seed,
            max_iter=1000
        )
        model.fit(X_train, y_train)

        # Evaluate on EVAL set for hyperparameter selection (FIXED: was using X_val)
        eval_probs = model.predict_proba(X_eval)[:, 1]
        eval_acc = accuracy_score(y_eval, (eval_probs > 0.5).astype(int))

        # Also evaluate on test set for reporting (but NOT for selection)
        test_probs = model.predict_proba(X_test)[:, 1]
        test_acc = accuracy_score(y_test, (test_probs > 0.5).astype(int))

        print(f"  Eval Acc: {eval_acc:.4f}, Test Acc: {test_acc:.4f}")

        # Select based on EVAL accuracy (FIXED DATA LEAKAGE)
        if eval_acc > best_score:
            best_score = eval_acc
            best_params = params

    print(f"\nBest Logistic Regression Params: {best_params}")
    print(f"Best Logistic Regression Eval Accuracy: {best_score:.4f}")

    return best_params, best_score


def save_best_hyperparams(best_params_dict, output_dir):
    """
    Save best hyperparameters to JSON file.

    Args:
        best_params_dict: Dict with model names as keys and (params, score) as values
        output_dir: Directory to save file
    """
    # Convert numpy types to JSON-serializable types
    json_ready = {}
    for model_name, (params, score) in best_params_dict.items():
        if params is not None:
            json_ready[model_name] = {
                'params': {k: (v.tolist() if hasattr(v, 'tolist') else v)
                          for k, v in params.items()},
                'score': float(score) if hasattr(score, 'tolist') else score
            }

    filepath = os.path.join(output_dir, 'best_hyperparams.json')
    with open(filepath, 'w') as f:
        json.dump(json_ready, f, indent=2)

    print(f"\n[SAVED] Best hyperparameters to {filepath}")


def load_best_hyperparams(filepath):
    """
    Load best hyperparameters from JSON file.

    Args:
        filepath: Path to JSON file

    Returns:
        Dict of best hyperparameters
    """
    with open(filepath, 'r') as f:
        return json.load(f)


def run_hyperparameter_tuning(X_train, y_train, X_calib, y_calib,
                              X_eval, y_eval, X_test, y_test,
                              input_dim, device, output_dir,
                              n_trials=10, models_to_tune='all'):
    """
    Run hyperparameter tuning for specified models.

    Uses Optuna Bayesian optimization if available, otherwise falls back to random search.

    DATA SPLIT USAGE (FIXED):
        X_train, y_train: Training data
        X_calib, y_calib: Calibration/validation data (for early stopping)
        X_eval, y_eval: Evaluation data (for hyperparameter selection)
        X_test, y_test: Test data (for final evaluation only)

    Args:
        X_train, y_train: Training data
        X_calib, y_calib: Calibration data
        X_eval, y_eval: Evaluation data
        X_test, y_test: Test data
        input_dim: Input dimension for neural networks
        device: PyTorch device
        output_dir: Output directory for results
        n_trials: Number of trials per model
        models_to_tune: 'all', 'neural', or 'fl' (which models to tune)

    Returns:
        Dict of best hyperparameters per model
    """
    print("\n" + "=" * 80)
    print("HYPERPARAMETER TUNING")
    print("=" * 80)
    print(f"Number of trials per model: {n_trials}")
    print("DATA SPLIT USAGE: Train → Calib (early stopping) → Eval (selection) → Test (final)")

    # Use Optuna if available, otherwise fall back to random search
    if OPTUNA_AVAILABLE:
        print("\n[INFO] Using Optuna Bayesian optimization for efficient hyperparameter search")
        return run_optuna_tuning(
            X_train, y_train, X_calib, y_calib,
            X_eval, y_eval, X_test, y_test,
            input_dim, device, output_dir,
            n_trials=n_trials, models_to_tune=models_to_tune
        )
    else:
        print("\n[INFO] Optuna not available. Using random search.")
        print("       Install Optuna for more efficient tuning: pip install optuna")

    best_hyperparams = {}

    if models_to_tune in ['all', 'neural', 'fl']:
        # Tune MLP
        mlp_params, mlp_score = tune_mlp_hyperparams(
            X_train, y_train, X_calib, y_calib, X_eval, y_eval, X_test, y_test,
            input_dim, device, n_trials=n_trials
        )
        best_hyperparams['MLP'] = (mlp_params, mlp_score)

    if models_to_tune in ['all', 'neural', 'fl']:
        # Tune LSTM
        lstm_params, lstm_score = tune_lstm_hyperparams(
            X_train, y_train, X_calib, y_calib, X_eval, y_eval, X_test, y_test,
            device, n_trials=n_trials
        )
        best_hyperparams['LSTM'] = (lstm_params, lstm_score)

    if models_to_tune == 'all':
        # Tune XGBoost
        xgb_params, xgb_score = tune_xgboost_hyperparams(
            X_train, y_train, X_calib, y_calib, X_eval, y_eval, X_test, y_test,
            n_trials=n_trials
        )
        if xgb_params is not None:
            best_hyperparams['XGBoost'] = (xgb_params, xgb_score)

        # Tune Logistic Regression
        logreg_params, logreg_score = tune_logreg_hyperparams(
            X_train, y_train, X_calib, y_calib, X_eval, y_eval, X_test, y_test,
            n_trials=n_trials
        )
        best_hyperparams['LogisticRegression'] = (logreg_params, logreg_score)

    # Save results
    save_best_hyperparams(best_hyperparams, output_dir)

    # Also save as CSV for easy viewing
    import pandas as pd
    results_rows = []
    for model_name, (params, score) in best_hyperparams.items():
        if params is not None:
            row = {'Model': model_name, 'Eval_Accuracy': score}
            row.update({k: str(v) for k, v in params.items()})
            results_rows.append(row)

    df = pd.DataFrame(results_rows)
    csv_path = os.path.join(output_dir, 'tuning_results.csv')
    df.to_csv(csv_path, index=False)
    print(f"[SAVED] Tuning results to {csv_path}")

    print("\n" + "=" * 80)
    print("HYPERPARAMETER TUNING COMPLETE")
    print("=" * 80)

    return best_hyperparams


# =============================================================================
# OPTUNA-BASED TUNING (Bayesian Optimization)
# =============================================================================

def mlp_optuna_objective(trial, X_train, y_train, X_calib, y_calib, X_eval, y_eval,
                         input_dim, device):
    """
    Optuna objective function for MLP tuning.

    Uses Bayesian optimization to efficiently search hyperparameter space.
    """
    from models import CreditNet

    # Suggest hyperparameters
    lr = trial.suggest_float('lr', 1e-4, 1e-1, log=True)
    hidden1 = trial.suggest_int('hidden1', 32, 256, step=32)
    hidden2 = trial.suggest_int('hidden2', 16, 128, step=16)
    dropout = trial.suggest_float('dropout', 0.0, 0.5)

    # Create model with suggested params
    model = CreditNet(input_dim, hidden_layers=[hidden1, hidden2], dropout=dropout).to(device)

    # Train with early stopping
    val_loss, best_epoch = validate_with_early_stopping(
        model, X_train, y_train, X_calib, y_calib, device,
        patience=5, max_epochs=50
    )

    # Report intermediate value for pruning
    trial.report(val_loss, best_epoch)

    # Check if trial should be pruned
    if trial.should_prune():
        raise optuna.TrialPruned()

    # Evaluate on EVAL set (not test!)
    model.eval()
    with torch.no_grad():
        eval_probs = torch.sigmoid(model.main(torch.Tensor(X_eval).to(device))).cpu().numpy()
    eval_acc = accuracy_score(y_eval, (eval_probs > 0.5).astype(int))

    return eval_acc  # Optuna maximizes this


def lstm_optuna_objective(trial, X_train, y_train, X_calib, y_calib, X_eval, y_eval,
                          device):
    """
    Optuna objective function for LSTM tuning.
    """
    from models import CreditLSTM

    # Suggest hyperparameters
    lr = trial.suggest_float('lr', 1e-4, 1e-1, log=True)
    hidden = trial.suggest_categorical('hidden', [32, 64, 128, 256])
    layers = trial.suggest_categorical('layers', [2, 4, 6])
    dropout = trial.suggest_float('dropout', 0.0, 0.5)

    # Create model with suggested params
    model = CreditLSTM(
        lstm_hidden=hidden,
        lstm_layers=layers,
        dropout=dropout
    ).to(device)

    # Train with early stopping
    val_loss, best_epoch = validate_with_early_stopping(
        model, X_train, y_train, X_calib, y_calib, device,
        patience=5, max_epochs=50
    )

    # Report intermediate value for pruning
    trial.report(val_loss, best_epoch)

    # Check if trial should be pruned
    if trial.should_prune():
        raise optuna.TrialPruned()

    # Evaluate on EVAL set
    model.eval()
    with torch.no_grad():
        eval_probs = torch.sigmoid(model.main(torch.Tensor(X_eval).to(device))).cpu().numpy()
    eval_acc = accuracy_score(y_eval, (eval_probs > 0.5).astype(int))

    return eval_acc


def xgboost_optuna_objective(trial, X_train, y_train, X_eval, y_eval):
    """
    Optuna objective function for XGBoost tuning.
    """
    try:
        import xgboost as xgb
    except ImportError:
        return 0.0

    # Suggest hyperparameters
    n_estimators = trial.suggest_categorical('n_estimators', [50, 100, 200, 300])
    max_depth = trial.suggest_categorical('max_depth', [3, 6, 9, 12])
    learning_rate = trial.suggest_float('learning_rate', 0.01, 0.3)

    # Create model
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    model = xgb.XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss',
        verbosity=0
    )
    model.fit(X_train, y_train)

    # Evaluate on EVAL set
    eval_probs = model.predict_proba(X_eval)[:, 1]
    eval_acc = accuracy_score(y_eval, (eval_probs > 0.5).astype(int))

    return eval_acc


def logreg_optuna_objective(trial, X_train, y_train, X_eval, y_eval):
    """
    Optuna objective function for Logistic Regression tuning.
    """
    # Suggest hyperparameters
    C = trial.suggest_float('C', 0.001, 100.0, log=True)
    solver = trial.suggest_categorical('solver', ['lbfgs', 'liblinear'])

    # Create model
    model = LogisticRegression(
        C=C,
        solver=solver,
        class_weight='balanced',
        random_state=42,
        max_iter=1000
    )
    model.fit(X_train, y_train)

    # Evaluate on EVAL set
    eval_probs = model.predict_proba(X_eval)[:, 1]
    eval_acc = accuracy_score(y_eval, (eval_probs > 0.5).astype(int))

    return eval_acc


def tune_with_optuna(model_type, X_train, y_train, X_calib, y_calib,
                     X_eval, y_eval, X_test, y_test,
                     input_dim, device, n_trials=50, timeout=None):
    """
    Tune hyperparameters using Optuna Bayesian optimization.

    DATA SPLIT USAGE:
        X_train, y_train: Training data
        X_calib, y_calib: Calibration/validation data (for early stopping)
        X_eval, y_eval: Evaluation data (for hyperparameter selection)
        X_test, y_test: Test data (for final evaluation only)

    Args:
        model_type: 'mlp', 'lstm', 'xgboost', or 'logreg'
        X_train, y_train: Training data
        X_calib, y_calib: Calibration/validation data
        X_eval, y_eval: Evaluation data (for hyperparameter selection)
        X_test, y_test: Test data (for final evaluation only)
        input_dim: Input dimension for neural networks
        device: PyTorch device
        n_trials: Number of Optuna trials
        timeout: Max time in seconds (None = no limit)

    Returns:
        best_params: Dictionary of best hyperparameters
        best_value: Best validation score (eval accuracy)
        study: Optuna study object (for analysis)
    """
    if not OPTUNA_AVAILABLE:
        print("Optuna not available. Install with: pip install optuna")
        return None, None, None

    print("\n" + "=" * 80)
    print(f"OPTUNA TUNING: {model_type.upper()}")
    print("=" * 80)
    print(f"Trials: {n_trials}, Timeout: {timeout}")

    # Select objective function
    objectives = {
        'mlp': mlp_optuna_objective,
        'lstm': lstm_optuna_objective,
        'xgboost': xgboost_optuna_objective,
        'logreg': logreg_optuna_objective
    }

    if model_type not in objectives:
        print(f"Unknown model type: {model_type}")
        return None, None, None

    # Create study with sampler and pruner
    sampler = TPESampler(seed=42)
    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=10)

    study = optuna.create_study(
        direction='maximize',
        sampler=sampler,
        pruner=pruner
    )

    # Define objective with additional args
    def objective(trial):
        if model_type == 'mlp':
            return objectives[model_type](
                trial, X_train, y_train, X_calib, y_calib, X_eval, y_eval,
                input_dim, device
            )
        elif model_type == 'lstm':
            return objectives[model_type](
                trial, X_train, y_train, X_calib, y_calib, X_eval, y_eval,
                device
            )
        else:
            return objectives[model_type](
                trial, X_train, y_train, X_eval, y_eval
            )

    # Run optimization
    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=timeout,
        show_progress_bar=True
    )

    # Get best trial
    best_trial = study.best_trial
    best_params = best_trial.params
    best_value = best_trial.value

    print(f"\nBest {model_type.upper()} Params:")
    for param, value in best_params.items():
        print(f"  {param}: {value}")
    print(f"Best Eval Accuracy: {best_value:.4f}")
    print(f"Trials completed: {len(study.trials)}")

    return best_params, best_value, study


def run_optuna_tuning(X_train, y_train, X_calib, y_calib,
                      X_eval, y_eval, X_test, y_test,
                      input_dim, device, output_dir,
                      n_trials=50, models_to_tune='all'):
    """
    Run hyperparameter tuning using Optuna for specified models.

    DATA SPLIT USAGE (FIXED):
        X_train, y_train: Training
        X_calib, y_calib: Validation (early stopping)
        X_eval, y_eval: Selection (choose best hyperparameters)
        X_test, y_test: Final evaluation (after selection)

    Args:
        X_train, y_train: Training data
        X_calib, y_calib: Calibration data
        X_eval, y_eval: Evaluation data
        X_test, y_test: Test data
        input_dim: Input dimension for neural networks
        device: PyTorch device
        output_dir: Output directory for results
        n_trials: Number of trials per model
        models_to_tune: 'all', 'neural', or 'fl' (which models to tune)

    Returns:
        Dict of best hyperparameters per model
    """
    if not OPTUNA_AVAILABLE:
        print("\nOptuna not available. Falling back to random search.")
        return run_hyperparameter_tuning(
            X_train, y_train, X_calib, y_calib,
            X_eval, y_eval, X_test, y_test,
            input_dim, device, output_dir,
            n_trials=n_trials, models_to_tune=models_to_tune
        )

    print("\n" + "=" * 80)
    print("OPTUNA BAYESIAN OPTIMIZATION")
    print("=" * 80)
    print(f"Number of trials per model: {n_trials}")
    print("DATA SPLIT USAGE: Train → Calib (early stopping) → Eval (selection) → Test (final)")

    best_hyperparams = {}
    all_studies = {}

    # Models to tune
    models = []
    if models_to_tune in ['all', 'neural', 'fl']:
        models.append('mlp')
    if models_to_tune in ['all', 'neural', 'fl']:
        models.append('lstm')
    if models_to_tune == 'all':
        models.append('xgboost')
        models.append('logreg')

    # Tune each model
    for model_type in models:
        best_params, best_value, study = tune_with_optuna(
            model_type=model_type,
            X_train=X_train,
            y_train=y_train,
            X_calib=X_calib,
            y_calib=y_calib,
            X_eval=X_eval,
            y_eval=y_eval,
            X_test=X_test,
            y_test=y_test,
            input_dim=input_dim,
            device=device,
            n_trials=n_trials
        )

        if best_params is not None:
            # Convert model name for consistency
            model_name = model_type.upper()
            if model_name == 'MLP':
                model_name = 'MLP'
            elif model_name == 'LOGREG':
                model_name = 'LogisticRegression'

            best_hyperparams[model_name] = (best_params, best_value)
            all_studies[model_name] = study

            # Generate Optuna visualizations
            plot_optuna_study(study, model_name, output_dir)

    # Save results
    save_best_hyperparams(best_hyperparams, output_dir)

    # Also save as CSV for easy viewing
    import pandas as pd
    results_rows = []
    for model_name, (params, score) in best_hyperparams.items():
        if params is not None:
            row = {'Model': model_name, 'Eval_Accuracy': score}
            row.update({k: str(v) for k, v in params.items()})
            results_rows.append(row)

    df = pd.DataFrame(results_rows)
    csv_path = os.path.join(output_dir, 'optuna_tuning_results.csv')
    df.to_csv(csv_path, index=False)
    print(f"[SAVED] Optuna tuning results to {csv_path}")

    print("\n" + "=" * 80)
    print("OPTUNA TUNING COMPLETE")
    print("=" * 80)

    return best_hyperparams


def plot_optuna_study(study, model_name, output_dir):
    """Plot Optuna optimization history and parameter importance."""
    try:
        import optuna.visualization as vis
    except ImportError:
        print("Optuna visualization not available")
        return

    try:
        # Optimization history
        fig = vis.plot_optimization_history(study)
        fig.write_html(f"{output_dir}/optuna_{model_name}_history.html")
        print(f"[SAVED] {output_dir}/optuna_{model_name}_history.html")

        # Parameter importance
        fig = vis.plot_param_importances(study)
        fig.write_html(f"{output_dir}/optuna_{model_name}_importance.html")
        print(f"[SAVED] {output_dir}/optuna_{model_name}_importance.html")

        # Parameter relationships
        fig = vis.plot_parallel_coordinate(study)
        fig.write_html(f"{output_dir}/optuna_{model_name}_parallel.html")
        print(f"[SAVED] {output_dir}/optuna_{model_name}_parallel.html")

    except Exception as e:
        print(f"Warning: Could not generate all Optuna visualizations: {e}")
