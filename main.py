#!/usr/bin/env python3
"""
Federated Learning Credit Scoring - Main Script

Run the complete federated learning experiment with:
- MLP and LSTM baselines
- Three-way calibration comparison (Platt, Temperature, FedCal)
- Comprehensive model comparison
- Fairness and explainability analysis

Usage:
    python main.py [--no-shap] [--no-lime] [--tune] [--tune-models MODELS]

    --no-shap:      Skip SHAP analysis (faster)
    --no-lime:      Skip LIME analysis (faster)
    --tune:         Run hyperparameter tuning before main experiment
    --tune-models:  Which models to tune: 'all', 'neural', 'fl', or 'none' (default: all)
"""

import os
import sys
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

# Import local modules
from config import (NUM_ROUNDS, get_output_dir, NUM_CLIENTS, UCI_FEATURE_NAMES,
                    TUNE_N_TRIALS, TUNE_PATIENCE, TUNE_MIN_DELTA, TUNE_MAX_EPOCHS,
                    RANDOM_STATE)
from data import load_data, create_splits, dirichlet_partition, get_tensors
from models import CreditNet, CreditLSTM
from federated import federated_training, train_client_mlp, train_client_lstm
from calibration import compare_calibration_methods, get_ece
from baselines import train_all_baselines
import visualization
import hyperparams

# Set reproducibility controls
def set_seed(seed=RANDOM_STATE):
    """Set random seeds for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Make PyTorch deterministic (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logging(output_dir):
    """Setup logging to file and stdout."""
    class Logger:
        def __init__(self, filepath):
            self.file = open(filepath, 'w', encoding='utf-8')

        def print(self, *args, **kwargs):
            print(*args, **kwargs)
            print(*args, file=self.file, **kwargs)
            self.file.flush()

    log_file = os.path.join(output_dir, "experiment_log.txt")
    logger = Logger(log_file)

    logger.print("=" * 80)
    logger.print("FEDERATED LEARNING EXPERIMENT - FULL RUN")
    logger.print("=" * 80)
    logger.print(f"Output Directory: {output_dir}")
    logger.print(f"Num Clients: {NUM_CLIENTS}")
    logger.print(f"Num Rounds: {NUM_ROUNDS}")
    logger.print("=" * 80)

    return logger


def plot_heterogeneity(client_indices, y_train, output_dir):
    """Plot client heterogeneity."""
    ratios = []
    for i in range(NUM_CLIENTS):
        y_local = y_train[client_indices[i]]
        dist = np.bincount(y_local)
        ratio = dist[1] / (dist[0] + dist[1]) if len(dist) > 1 else 0
        ratios.append(ratio)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(range(NUM_CLIENTS), ratios, color='steelblue', alpha=0.7)
    ax.axhline(y=np.mean(ratios), color='red', linestyle='--', label='Mean')
    ax.set_xlabel('Client ID')
    ax.set_ylabel('Default Ratio')
    ax.set_title('Data Heterogeneity Across Clients (Dirichlet α=0.5)')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '01_client_heterogeneity.png'), dpi=150)
    plt.close()


def plot_convergence(test_accs, model_name, output_dir, filename):
    """Plot training convergence."""
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(range(1, len(test_accs) + 1), test_accs, marker='o', color='steelblue')
    ax.set_xlabel('Round')
    ax.set_ylabel('Test Accuracy')
    ax.set_title(f'{model_name} Convergence')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), dpi=150)
    plt.close()


def plot_model_comparison(results, output_dir):
    """Plot comprehensive model comparison."""
    models = list(results.keys())
    metrics = ['Accuracy', 'AUC', 'F1']

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('Model Comparison: Performance Metrics', fontsize=16, fontweight='bold')

    for idx, metric in enumerate(metrics):
        values = [results[m].get(metric.lower(), 0) for m in models]
        colors = ['lightgray'] * (len(models) - 1) + ['steelblue']

        axes[idx].bar(models, values, color=colors, alpha=0.8, edgecolor='black')
        axes[idx].set_ylabel(metric)
        axes[idx].set_title(f'{metric} Comparison')
        axes[idx].set_ylim(min(values) * 0.9, max(values) * 1.1)
        axes[idx].grid(axis='y', alpha=0.3)
        for tick in axes[idx].get_xticklabels():
            tick.set_rotation(15)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '04_model_comparison.png'), dpi=150)
    plt.close()


def evaluate_model(model, X_test, y_test, scaler=None, device='cpu'):
    """Evaluate model on test set."""
    model.eval()
    with torch.no_grad():
        logits = model.main(torch.Tensor(X_test).to(device))
        raw_probs = torch.sigmoid(logits).cpu().numpy().flatten()

    if scaler is not None:
        probs = scaler(logits).detach().cpu().numpy().flatten()
    else:
        probs = raw_probs

    preds = (probs > 0.5).astype(int)

    return {
        'accuracy': accuracy_score(y_test, preds),
        'auc': roc_auc_score(y_test, probs),
        'f1': f1_score(y_test, preds, zero_division=0),
        'ece': get_ece(probs, y_test)
    }


def main(args):
    """Main experiment runner."""
    # Setup reproducibility
    set_seed()
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = get_output_dir()
    logger = setup_logging(output_dir)

    logger.print(f"\nUsing device: {device}")
    logger.print(f"Random seed: {RANDOM_STATE}")

    # =========================================================================
    # STEP 1: Load Data
    # =========================================================================
    logger.print("\n" + "=" * 80)
    logger.print("STEP 1: Loading Data")
    logger.print("=" * 80)

    X, y, feature_names = load_data()

    # =========================================================================
    # STEP 2: Create Splits
    # =========================================================================
    logger.print("\n" + "=" * 80)
    logger.print("STEP 2: Creating Splits (Scaler fit on train only)")
    logger.print("=" * 80)

    # Note: create_splits now fits scaler on train data only and saves it
    X_train, X_calib, X_eval, X_test, y_train, y_calib, y_eval, y_test = create_splits(X, y, output_dir=output_dir)



    # =========================================================================
    # STEP 3: Dirichlet Partitioning
    # =========================================================================
    logger.print("\n" + "=" * 80)
    logger.print("STEP 3: Dirichlet Partitioning (Non-IID)")
    logger.print("=" * 80)

    client_indices = dirichlet_partition(y_train)
    plot_heterogeneity(client_indices, y_train, output_dir)
    logger.print(f"[SAVED] {output_dir}/01_client_heterogeneity.png")

    # =========================================================================
    # STEP 3.5: Hyperparameter Tuning (Optional)
    # =========================================================================
    best_hyperparams = None
    if args.tune and args.tune_models != 'none':
        logger.print("\n" + "=" * 80)
        logger.print("STEP 3.5: Hyperparameter Tuning")
        logger.print("=" * 80)
        logger.print(f"Models to tune: {args.tune_models}")
        logger.print(f"Trials per model: {args.tune_trials}")

        best_hyperparams = hyperparams.run_hyperparameter_tuning(
            X_train=X_train,
            y_train=y_train,
            X_calib=X_calib,
            y_calib=y_calib,
            X_eval=X_eval,
            y_eval=y_eval,
            X_test=X_test,
            y_test=y_test,
            input_dim=X.shape[1],
            device=device,
            output_dir=output_dir,
            n_trials=args.tune_trials,
            models_to_tune=args.tune_models
        )

        logger.print(f"\n[COMPLETE] Hyperparameter tuning results saved to {output_dir}/")
    else:
        logger.print("\n" + "-" * 80)
        logger.print("Skipping hyperparameter tuning (use --tune to enable)")
        logger.print("-" * 80)

    # Load tuned hyperparameters from JSON if they exist
    # This allows using previously tuned parameters without re-running tuning
    tuned_params_file = os.path.join(output_dir, 'best_hyperparams.json')
    if os.path.exists(tuned_params_file):

        with open(tuned_params_file, 'r') as f:
            all_tuned_params = json.load(f)
        logger.print(f"\n[LOADING] Tuned hyperparameters from {tuned_params_file}")

        # Print loaded parameters for each model
        for model_name, params in all_tuned_params.items():
            logger.print(f"  {model_name}: {params.get('params', {})}")
    else:
        all_tuned_params = None

    # =========================================================================
    # STEP 4: Federated Learning with MLP
    # =========================================================================
    logger.print("\n" + "=" * 80)
    logger.print("STEP 4: Federated Learning (MLP)")
    logger.print("=" * 80)

    # Get tuned MLP parameters or use defaults
    # Note: Hyperparams are saved with key 'MLP', not 'FL-FedAvg (MLP)'
    mlp_params = all_tuned_params.get('MLP', {}).get('params', {}) if all_tuned_params else {}

    # Build model kwargs with dynamic architecture
    mlp_kwargs = {'input_dim': X.shape[1]}
    if mlp_params:
        if 'hidden1' in mlp_params and 'hidden2' in mlp_params:
            mlp_kwargs['hidden_layers'] = [mlp_params['hidden1'], mlp_params['hidden2']]
        if 'dropout' in mlp_params:
            mlp_kwargs['dropout'] = mlp_params['dropout']

    mlp_model, mlp_accs = federated_training(
        model_class=CreditNet,
        model_kwargs=mlp_kwargs,
        train_fn=train_client_mlp,
        X_train=X_train,
        y_train=y_train,
        client_indices=client_indices,
        X_test=X_test,
        y_test=y_test,
        device=device,
        num_rounds=NUM_ROUNDS,
        model_name="FL-FedAvg (MLP)",
        hyperparams=mlp_params if mlp_params else None
    )

    # plot_convergence(mlp_accs, "FL-FedAvg (MLP)", output_dir, '02_fl_convergence.png')
    # logger.print(f"[SAVED] {output_dir}/02_fl_convergence.png")
    
    # Save MLP model and hyperparameters
    torch.save(mlp_model.state_dict(), f"{output_dir}/mlp_model.pth")
    logger.print(f"[SAVED] {output_dir}/mlp_model.pth")
    
    mlp_config = {
        'model_class': 'CreditNet',
        'model_kwargs': mlp_kwargs,
        'hyperparameters': mlp_params if mlp_params else {},
        'final_accuracy': mlp_accs[-1],
        'convergence_history': mlp_accs
    }
    with open(f"{output_dir}/mlp_config.json", 'w') as f:
        json.dump(mlp_config, f, indent=2)
    logger.print(f"[SAVED] {output_dir}/mlp_config.json")

    # =========================================================================
    # STEP 5: Calibration Methods Comparison
    # =========================================================================
    logger.print("\n" + "=" * 80)
    logger.print("STEP 5: Calibration Methods Comparison")
    logger.print("=" * 80)

    calib_results = compare_calibration_methods(
        mlp_model, X_calib, y_calib, X_eval, y_eval,
        client_indices, X_train, y_train, device
    )

    # Calibration Visualization (4-panel comparison)
    # Visualization is handled in Step 7b for all models
    # visualization.plot_calibration_comparison(calib_results['results'], output_dir)
    
    # Save calibration models
    torch.save(calib_results['scalers']['platt'].state_dict(), f"{output_dir}/platt_scaler.pth")
    torch.save(calib_results['scalers']['temp'].state_dict(), f"{output_dir}/temperature_scaler.pth")
    torch.save(calib_results['scalers']['beta'].state_dict(), f"{output_dir}/beta_scaler.pth")
    logger.print(f"[SAVED] Calibration models: platt_scaler.pth, temperature_scaler.pth, beta_scaler.pth")

    # Before/After calibration for all models
    # Note: We'll add Central NN and LSTM after they're trained
    # Compute before/after metrics for visualization
    before_after_results = {}
    for model_name, model_obj in [
        ('FL-FedAvg (MLP)', mlp_model),
    ]:
        model_obj.eval()
        test_logits = model_obj.main(torch.Tensor(X_test).to(device)).detach()

        # Uncalibrated metrics
        raw_probs = torch.sigmoid(test_logits).cpu().numpy().flatten()
        raw_preds = (raw_probs > 0.5).astype(int)
        raw_acc = accuracy_score(y_test, raw_preds)
        raw_f1 = f1_score(y_test, raw_preds, zero_division=0)
        raw_ece = get_ece(raw_probs, y_test)

        # Calibrated metrics (using FedCal)
        cal_probs = calib_results['scalers']['fedcal'](test_logits).detach().cpu().numpy().flatten()
        cal_preds = (cal_probs > 0.5).astype(int)
        cal_acc = accuracy_score(y_test, cal_preds)
        cal_f1 = f1_score(y_test, cal_preds, zero_division=0)
        cal_ece = get_ece(cal_probs, y_test)

        before_after_results[model_name] = {
            'before': {'accuracy': raw_acc, 'f1': raw_f1, 'ece': raw_ece},
            'after': {'accuracy': cal_acc, 'f1': cal_f1, 'ece': cal_ece}
        }

    # Add Local NN (representative from client 0)
    local_model_0 = CreditNet(X_train.shape[1]).to(device)
    idx_0 = client_indices[0]
    local_model_0.train()
    local_opt = torch.optim.Adam(local_model_0.parameters(), lr=0.005)
    local_criterion = nn.BCELoss()
    from torch.utils.data import DataLoader, TensorDataset
    local_ds = TensorDataset(
        torch.Tensor(X_train[idx_0]),
        torch.Tensor(y_train[idx_0]).float().unsqueeze(1)
    )
    local_dl = DataLoader(local_ds, batch_size=64, shuffle=True, drop_last=True)
    for epoch in range(15):
        for bx, by in local_dl:
            bx, by = bx.to(device), by.to(device)
            local_opt.zero_grad()
            loss = local_criterion(local_model_0(bx), by)
            loss.backward()
            local_opt.step()

    local_model_0.eval()
    local_test_logits = local_model_0.main(torch.Tensor(X_test).to(device)).detach()
    local_raw_probs = torch.sigmoid(local_test_logits).cpu().numpy().flatten()
    local_raw_preds = (local_raw_probs > 0.5).astype(int)
    local_raw_acc = accuracy_score(y_test, local_raw_preds)
    local_raw_f1 = f1_score(y_test, local_raw_preds, zero_division=0)
    local_raw_ece = get_ece(local_raw_probs, y_test)

    # Use FedCal for local (note: this is not ideal, but for comparison)
    local_cal_probs = calib_results['scalers']['fedcal'](local_test_logits).detach().cpu().numpy().flatten()
    local_cal_preds = (local_cal_probs > 0.5).astype(int)
    local_cal_acc = accuracy_score(y_test, local_cal_preds)
    local_cal_f1 = f1_score(y_test, local_cal_preds, zero_division=0)
    local_cal_ece = get_ece(local_cal_probs, y_test)

    before_after_results['Local NN (Client 0)'] = {
        'before': {'accuracy': local_raw_acc, 'f1': local_raw_f1, 'ece': local_raw_ece},
        'after': {'accuracy': local_cal_acc, 'f1': local_cal_f1, 'ece': local_cal_ece}
    }

    # Note: before_after_results will be completed and plotted after Central NN and LSTM are trained

    # =========================================================================
    # STEP 6: Federated Learning with LSTM
    # =========================================================================
    logger.print("\n" + "=" * 80)
    logger.print("STEP 6: Federated Learning (LSTM)")
    logger.print("=" * 80)

    # Get tuned LSTM parameters or use defaults
    # Note: Hyperparams are saved with key 'LSTM', not 'FL-FedAvg (LSTM)'
    lstm_params = all_tuned_params.get('LSTM', {}).get('params', {}) if all_tuned_params else {}

    # Build model kwargs with dynamic architecture
    lstm_kwargs = {}
    if lstm_params:
        if 'hidden' in lstm_params:
            lstm_kwargs['lstm_hidden'] = lstm_params['hidden']
        if 'layers' in lstm_params:
            lstm_kwargs['lstm_layers'] = lstm_params['layers']
        if 'dropout' in lstm_params:
            lstm_kwargs['dropout'] = lstm_params['dropout']

    lstm_model, lstm_accs = federated_training(
        model_class=CreditLSTM,
        model_kwargs=lstm_kwargs,
        train_fn=train_client_lstm,
        X_train=X_train,
        y_train=y_train,
        client_indices=client_indices,
        X_test=X_test,
        y_test=y_test,
        device=device,
        num_rounds=NUM_ROUNDS,
        model_name="FL-FedAvg (LSTM)",
        hyperparams=lstm_params if lstm_params else None
    )

    # plot_convergence(lstm_accs, "FL-FedAvg (LSTM)", output_dir, '03_lstm_convergence.png')
    # logger.print(f"[SAVED] {output_dir}/03_lstm_convergence.png")
    
    # Save LSTM model and hyperparameters
    torch.save(lstm_model.state_dict(), f"{output_dir}/lstm_model.pth")
    logger.print(f"[SAVED] {output_dir}/lstm_model.pth")
    
    lstm_config = {
        'model_class': 'CreditLSTM',
        'model_kwargs': lstm_kwargs,
        'hyperparameters': lstm_params if lstm_params else {},
        'final_accuracy': lstm_accs[-1],
        'convergence_history': lstm_accs
    }
    with open(f"{output_dir}/lstm_config.json", 'w') as f:
        json.dump(lstm_config, f, indent=2)
    logger.print(f"[SAVED] {output_dir}/lstm_config.json")

    # =========================================================================
    # STEP 7: Train Baselines
    # =========================================================================
    logger.print("\n" + "=" * 80)
    logger.print("STEP 7: Training Baseline Models")
    logger.print("=" * 80)

    baseline_results, central_model = train_all_baselines(
        X_train, y_train, X_test, y_test,
        client_indices, X.shape[1], device,
        tuned_params=all_tuned_params  # Pass tuned parameters to baselines
    )

    # Save Central NN model
    if central_model:
        torch.save(central_model.state_dict(), f"{output_dir}/central_nn.pth")
        logger.print(f"[SAVED] {output_dir}/central_nn.pth")

    # =========================================================================
    # STEP 7b: Comprehensive Calibration Analysis (All Models)
    # =========================================================================
    logger.print("\n" + "=" * 80)
    logger.print("STEP 7b: Comprehensive Calibration Analysis")
    logger.print("=" * 80)
    
    # Collect all neural models for calibration
    all_models = {
        'FL-FedAvg (MLP)': mlp_model,
        'FL-FedAvg (LSTM)': lstm_model
    }
    if central_model:
        all_models['Central NN'] = central_model
        
    import calibration_analysis
    
    # Run calibration on all models
    calib_df, calib_models = calibration_analysis.calibrate_all_models(
        all_models, X_calib, y_calib, X_eval, y_eval, device
    )
    
    # Save visualizations
    calibration_analysis.plot_calibration_heatmap(calib_df, output_dir)
    # calibration_analysis.plot_calibration_per_method(calib_df, output_dir)
    
    # Plot Reliability Diagrams (Fig2)
    visualization.plot_calibration_summary(
        all_models, calib_models, X_eval, y_eval, device, output_dir
    )
    
    # Save CSVs
    calibration_analysis.save_calibration_results_csv(calib_df, output_dir)
    
    logger.print("\nCalibration Analysis Summary (Best ECE per model):")
    for model_name in all_models:
        best_ece = calib_df[calib_df['Model'] == model_name].sort_values('ECE').iloc[0]
        logger.print(f"  {model_name}: {best_ece['Calibration']} (ECE={best_ece['ECE']:.4f})")

    # =========================================================================
    # STEP 8: Final Evaluation & Comparison
    # =========================================================================
    logger.print("\n" + "=" * 80)
    logger.print("STEP 8: Final Model Comparison")
    logger.print("=" * 80)

    # Compile final results using best calibration for each model
    final_results_list = []
    
    # 1. Add Non-Neural Baselines
    for name, metrics in baseline_results.items():
        if name == 'Logistic Regression' or name.startswith('Local'):
             final_results_list.append({
                 'Model': name,
                 'Accuracy': metrics['accuracy'],
                 'F1': metrics['f1'] if 'f1' in metrics else 0,
                 'ECE': metrics['ece']
             })

    # 2. Add Neural Models (FL-MLP, FL-LSTM, Central NN) - Picking Best Calibration
    for model_name in all_models:
        # Pick the version with best ECE
        best_row = calib_df[calib_df['Model'] == model_name].sort_values('ECE').iloc[0]
        
        final_results_list.append({
            'Model': model_name,
            'Accuracy': best_row['Accuracy'],
            'F1': best_row['F1'],
            'ECE': best_row['ECE']
        })
        
    final_df = pd.DataFrame(final_results_list)
    final_df.to_csv(f"{output_dir}/final_model_comparison.csv", index=False)
    
    # Plot final comparison (Fig3)
    visualization.plot_final_comparison(final_df, output_dir)
    
    # Plot combined convergence (Fig1)
    if 'mlp_accs' in locals() and 'lstm_accs' in locals():
        visualization.plot_convergence(mlp_accs, lstm_accs, output_dir)
    
    # Print summary table
    logger.print("\nFINAL MODEL LEADERBOARD:")
    logger.print(final_df.sort_values('F1', ascending=False).to_string(index=False))




    # =========================================================================
    # STEP 9: Explainability (SHAP & LIME)
    # =========================================================================
    if not args.no_shap:
        try:
            import shap
            logger.print("\n" + "=" * 80)
            logger.print("STEP 9: SHAP Analysis")
            logger.print("=" * 80)

            def predict_fn(x_numpy):
                mlp_model.eval()
                with torch.no_grad():
                    tensor_x = torch.Tensor(x_numpy).to(device)
                    logits = mlp_model.main(tensor_x)
                    probs = torch.sigmoid(logits).cpu().numpy().flatten()
                    return np.vstack([1 - probs, probs]).T

            background_data = shap.kmeans(X_train, 10)
            explainer = shap.KernelExplainer(lambda x: predict_fn(x)[:, 1], background_data)
            shap_values = explainer.shap_values(X_test[:50], nsamples=100)

            fig = plt.figure(figsize=(10, 6))
            shap.summary_plot(shap_values, X_test[:50], show=False)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, '07_shap_summary.png'), dpi=150, bbox_inches='tight')
            plt.close()
            logger.print(f"[SAVED] {output_dir}/07_shap_summary.png")
        except Exception as e:
            logger.print(f"SHAP error: {e}")

    # LIME Explanation
    if not args.no_lime:
        try:
            logger.print("\n" + "=" * 80)
            logger.print("STEP 9b: LIME Analysis")
            logger.print("=" * 80)

            visualization.generate_lime_explanation(
                model=mlp_model,
                X_train=X_train,
                X_test=X_test,
                y_test=y_test,
                feature_names=UCI_FEATURE_NAMES,
                sample_idx=5,
                output_dir=output_dir
            )
        except Exception as e:
            logger.print(f"LIME error: {e}")

    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    logger.print("\n" + "=" * 80)
    logger.print("EXPERIMENT COMPLETE")
    logger.print("=" * 80)
    logger.print(f"\nResults saved to: {output_dir}/")
    logger.print("  - experiment_log.txt")
    logger.print("  - results_summary.csv")
    logger.print("  - Visualization figures")

    # Key findings
    logger.print("\n" + "=" * 80)
    logger.print("KEY FINDINGS")
    logger.print("=" * 80)

    logger.print("  - Results Summary and Leaderboard above")

    logger.print("\n" + "=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Federated Learning Credit Scoring')
    parser.add_argument('--no-shap', action='store_true', help='Skip SHAP analysis')
    parser.add_argument('--no-lime', action='store_true', help='Skip LIME analysis')
    parser.add_argument('--tune', action='store_true', help='Run hyperparameter tuning')
    parser.add_argument('--tune-models', type=str, default='all',
                        choices=['all', 'neural', 'fl', 'none'],
                        help='Which models to tune (default: all)')
    parser.add_argument('--tune-trials', type=int, default=TUNE_N_TRIALS,
                        help=f'Number of tuning trials per model (default: {TUNE_N_TRIALS})')
    args = parser.parse_args()

    main(args)
