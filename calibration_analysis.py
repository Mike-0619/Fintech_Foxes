"""
Comprehensive calibration analysis across all models.

This module extends the calibration comparison to work with multiple models
and generates detailed visualizations and CSV reports.
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score

from calibration import platt_scaling, temperature_scaling, beta_scaling, get_ece
from models import PlattScaler, TemperatureScaler, BetaCalibrator
import torch.nn as nn
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss


def calibrate_all_models(models_dict, X_calib, y_calib, X_eval, y_eval, device='cpu'):
    """
    Apply all calibration methods to all models.
    
    Args:
        models_dict: Dict of {model_name: model}
        X_calib: Calibration features
        y_calib: Calibration labels
        X_eval: Evaluation features
        y_eval: Evaluation labels
        device: PyTorch device
        
    Returns:
        results_df: DataFrame with all combinations
        calibrators_dict: Dict of {(model_name, calib_method): calibrator}
    """
    results = []
    calibrators_dict = {}
    
    calibration_methods = ['Uncalibrated', 'Platt', 'Temperature', 'Beta']
    
    for model_name, model in models_dict.items():
        model.eval()
        
        # Get logits on calibration and evaluation sets
        with torch.no_grad():
            calib_logits = model.main(torch.Tensor(X_calib).to(device)).cpu()
            eval_logits = model.main(torch.Tensor(X_eval).to(device)).cpu()
        
        # 1. Uncalibrated
        uncalib_probs = torch.sigmoid(eval_logits).numpy().flatten()
        uncalib_preds = (uncalib_probs > 0.5).astype(int)
        
        results.append({
            'Model': model_name,
            'Calibration': 'Uncalibrated',
            'Accuracy': accuracy_score(y_eval, uncalib_preds),
            'F1': f1_score(y_eval, uncalib_preds, zero_division=0),
            'ECE': get_ece(uncalib_probs, y_eval),
            'Brier': brier_score_loss(y_eval, uncalib_probs)
        })
        
        # 2. Platt Scaling
        # Train PlattScaler directly using pre-computed logits
        platt_scaler = PlattScaler().to(device)
        optimizer = torch.optim.LBFGS([platt_scaler.A, platt_scaler.B], lr=0.01, max_iter=100)
        
        calib_logits_dev = calib_logits.to(device)
        calib_labels_dev = torch.Tensor(y_calib).float().unsqueeze(1).to(device)
        
        def platt_closure():
            optimizer.zero_grad()
            loss = torch.nn.BCEWithLogitsLoss()(platt_scaler(calib_logits_dev), calib_labels_dev)
            loss.backward()
            return loss
        
        optimizer.step(platt_closure)
        calibrators_dict[(model_name, 'Platt')] = platt_scaler
        
        platt_scaler.eval()
        with torch.no_grad():
            platt_probs = platt_scaler(eval_logits.to(device)).cpu().numpy().flatten()
        platt_preds = (platt_probs > 0.5).astype(int)
        
        results.append({
            'Model': model_name,
            'Calibration': 'Platt',
            'Accuracy': accuracy_score(y_eval, platt_preds),
            'F1': f1_score(y_eval, platt_preds, zero_division=0),
            'ECE': get_ece(platt_probs, y_eval),
            'Brier': brier_score_loss(y_eval, platt_probs)
        })
        
        # 3. Temperature Scaling
        temp_scaler = TemperatureScaler().to(device)
        optimizer = torch.optim.LBFGS([temp_scaler.temperature], lr=0.01, max_iter=100)
        
        def temp_closure():
            optimizer.zero_grad()
            loss = torch.nn.BCEWithLogitsLoss()(temp_scaler(calib_logits_dev), calib_labels_dev)
            loss.backward()
            return loss
            
        optimizer.step(temp_closure)
        calibrators_dict[(model_name, 'Temperature')] = temp_scaler
        
        temp_scaler.eval()
        with torch.no_grad():
            temp_probs = temp_scaler(eval_logits.to(device)).cpu().numpy().flatten()
        temp_preds = (temp_probs > 0.5).astype(int)
        
        results.append({
            'Model': model_name,
            'Calibration': 'Temperature',
            'Accuracy': accuracy_score(y_eval, temp_preds),
            'F1': f1_score(y_eval, temp_preds, zero_division=0),
            'ECE': get_ece(temp_probs, y_eval),
            'Brier': brier_score_loss(y_eval, temp_probs)
        })
        
        # 4. Beta Calibration
        beta_scaler = BetaCalibrator().to(device)
        optimizer = torch.optim.LBFGS([beta_scaler.a, beta_scaler.b, beta_scaler.c], lr=0.01, max_iter=100)
        
        def beta_closure():
            optimizer.zero_grad()
            loss = torch.nn.BCEWithLogitsLoss()(beta_scaler(calib_logits_dev), calib_labels_dev)
            loss.backward()
            return loss
            
        optimizer.step(beta_closure)
        calibrators_dict[(model_name, 'Beta')] = beta_scaler
        
        beta_scaler.eval()
        with torch.no_grad():
            beta_probs = beta_scaler(eval_logits.to(device)).cpu().numpy().flatten()
        beta_preds = (beta_probs > 0.5).astype(int)
        
        results.append({
            'Model': model_name,
            'Calibration': 'Beta',
            'Accuracy': accuracy_score(y_eval, beta_preds),
            'F1': f1_score(y_eval, beta_preds, zero_division=0),
            'ECE': get_ece(beta_probs, y_eval),
            'Brier': brier_score_loss(y_eval, beta_probs)
        })
    
    results_df = pd.DataFrame(results)
    return results_df, calibrators_dict


def plot_reliability_diagrams(models_dict, calibrators_dict, X_eval, y_eval, output_dir, device='cpu'):
    """
    Plot Reliability Diagrams (Calibration Curves) for all models and methods.
    
    Args:
        models_dict: Dict of base models
        calibrators_dict: Dict of {(model_name, method): calibrator}
        X_eval: Evaluation features
        y_eval: Evaluation labels
        output_dir: Directory to save plots
        device: Device to run on
    """
    for model_name, model in models_dict.items():
        plt.figure(figsize=(10, 8))
        
        # Get base logits
        model.eval()
        with torch.no_grad():
            logits = model.main(torch.Tensor(X_eval).to(device)).cpu()
            
        # 1. Uncalibrated
        probs = torch.sigmoid(logits).numpy().flatten()
        fraction_of_positives, mean_predicted_value = calibration_curve(y_eval, probs, n_bins=10)
        plt.plot(mean_predicted_value, fraction_of_positives, "s-", label=f'Uncalibrated (Brier={brier_score_loss(y_eval, probs):.4f})')
        
        # 2. Calibrated methods
        methods = ['Platt', 'Temperature', 'Beta']
        markers = ['o-', '^-', 'x-']
        
        for method, marker in zip(methods, markers):
            if (model_name, method) in calibrators_dict:
                calibrator = calibrators_dict[(model_name, method)]
                calibrator.eval()
                with torch.no_grad():
                    calib_probs = calibrator(logits.to(device)).cpu().numpy().flatten()
                
                frac_pos, mean_pred = calibration_curve(y_eval, calib_probs, n_bins=10)
                brier = brier_score_loss(y_eval, calib_probs)
                plt.plot(mean_pred, frac_pos, marker, label=f'{method} (Brier={brier:.4f})')

        plt.plot([0, 1], [0, 1], "k:", label="Perfectly Calibrated")
        plt.ylabel("Fraction of positives")
        plt.xlabel("Mean predicted probability")
        plt.title(f'Reliability Diagram: {model_name}')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        safe_name = model_name.replace(" ", "_").replace("-", "").replace("(", "").replace(")", "").lower()
        plt.savefig(f"{output_dir}/reliability_{safe_name}.png", dpi=150)
        plt.close()
        print(f"[SAVED] Reliability diagram for {model_name}")


def plot_calibration_per_method(results_df, output_dir):
    """
    Create one plot per calibration method showing all models.
    
    Args:
        results_df: DataFrame from calibrate_all_models
        output_dir: Directory to save plots
    """
    calibration_methods = results_df['Calibration'].unique()
    
    for calib_method in calibration_methods:
        method_data = results_df[results_df['Calibration'] == calib_method]
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(f'{calib_method} Calibration - All Models', fontsize=16, fontweight='bold')
        
        models = method_data['Model'].values
        x = np.arange(len(models))
        width = 0.6
        
        # Accuracy
        axes[0].bar(x, method_data['Accuracy'].values, width, color='steelblue', alpha=0.8)
        axes[0].set_ylabel('Accuracy', fontsize=12)
        axes[0].set_title('Accuracy Comparison', fontsize=12)
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(models, rotation=15, ha='right')
        axes[0].grid(axis='y', alpha=0.3)
        axes[0].set_ylim([0, 1])
        
        # F1 Score
        axes[1].bar(x, method_data['F1'].values, width, color='coral', alpha=0.8)
        axes[1].set_ylabel('F1 Score', fontsize=12)
        axes[1].set_title('F1 Score Comparison', fontsize=12)
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(models, rotation=15, ha='right')
        axes[1].grid(axis='y', alpha=0.3)
        axes[1].set_ylim([0, 1])
        
        # ECE
        axes[2].bar(x, method_data['ECE'].values, width, color='mediumseagreen', alpha=0.8)
        axes[2].set_ylabel('ECE (Lower is Better)', fontsize=12)
        axes[2].set_title('Calibration Error', fontsize=12)
        axes[2].set_xticks(x)
        axes[2].set_xticklabels(models, rotation=15, ha='right')
        axes[2].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        # Save with safe filename
        safe_name = calib_method.lower().replace(' ', '_')
        filepath = f'{output_dir}/calibration_{safe_name}_all_models.png'
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"[SAVED] {filepath}")


def plot_calibration_heatmap(results_df, output_dir):
    """
    Create heatmap showing ECE for all model-calibration combinations.
    
    Args:
        results_df: DataFrame from calibrate_all_models
        output_dir: Directory to save plot
    """
    # Pivot to create heatmap data
    pivot_acc = results_df.pivot(index='Model', columns='Calibration', values='Accuracy')
    pivot_f1 = results_df.pivot(index='Model', columns='Calibration', values='F1')
    pivot_ece = results_df.pivot(index='Model', columns='Calibration', values='ECE')
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Calibration Methods Comparison - All Models', fontsize=16, fontweight='bold')
    
    # Accuracy heatmap
    im1 = axes[0].imshow(pivot_acc.values, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    axes[0].set_xticks(np.arange(len(pivot_acc.columns)))
    axes[0].set_yticks(np.arange(len(pivot_acc.index)))
    axes[0].set_xticklabels(pivot_acc.columns, rotation=45, ha='right')
    axes[0].set_yticklabels(pivot_acc.index)
    axes[0].set_title('Accuracy (Higher is Better)')
    
    # Add text annotations
    for i in range(len(pivot_acc.index)):
        for j in range(len(pivot_acc.columns)):
            text = axes[0].text(j, i, f'{pivot_acc.values[i, j]:.3f}',
                               ha="center", va="center", color="black", fontsize=9)
    
    plt.colorbar(im1, ax=axes[0])
    
    # F1 heatmap
    im2 = axes[1].imshow(pivot_f1.values, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    axes[1].set_xticks(np.arange(len(pivot_f1.columns)))
    axes[1].set_yticks(np.arange(len(pivot_f1.index)))
    axes[1].set_xticklabels(pivot_f1.columns, rotation=45, ha='right')
    axes[1].set_yticklabels(pivot_f1.index)
    axes[1].set_title('F1 Score (Higher is Better)')
    
    for i in range(len(pivot_f1.index)):
        for j in range(len(pivot_f1.columns)):
            text = axes[1].text(j, i, f'{pivot_f1.values[i, j]:.3f}',
                               ha="center", va="center", color="black", fontsize=9)
    
    plt.colorbar(im2, ax=axes[1])
    
    # ECE heatmap
    im3 = axes[2].imshow(pivot_ece.values, cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=0.3)
    axes[2].set_xticks(np.arange(len(pivot_ece.columns)))
    axes[2].set_yticks(np.arange(len(pivot_ece.index)))
    axes[2].set_xticklabels(pivot_ece.columns, rotation=45, ha='right')
    axes[2].set_yticklabels(pivot_ece.index)
    axes[2].set_title('ECE (Lower is Better)')
    
    for i in range(len(pivot_ece.index)):
        for j in range(len(pivot_ece.columns)):
            text = axes[2].text(j, i, f'{pivot_ece.values[i, j]:.3f}',
                               ha="center", va="center", color="black", fontsize=9)
    
    plt.colorbar(im3, ax=axes[2])
    
    plt.tight_layout()
    filepath = f'{output_dir}/calibration_heatmap_all_models.png'
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"[SAVED] {filepath}")


def save_calibration_results_csv(results_df, output_dir):
    """
    Save detailed calibration results to CSV.
    
    Args:
        results_df: DataFrame from calibrate_all_models
        output_dir: Directory to save CSV
    """
    filepath = f'{output_dir}/calibration_detailed_results.csv'
    results_df.to_csv(filepath, index=False, float_format='%.6f')
    print(f"[SAVED] {filepath}")
    
    # Also create a summary showing best calibration per model
    summary = []
    for model in results_df['Model'].unique():
        model_data = results_df[results_df['Model'] == model]
        
        # Best for each metric
        best_acc = model_data.loc[model_data['Accuracy'].idxmax()]
        best_f1 = model_data.loc[model_data['F1'].idxmax()]
        best_ece = model_data.loc[model_data['ECE'].idxmin()]
        best_brier = model_data.loc[model_data['Brier'].idxmin()]
        
        summary.append({
            'Model': model,
            'Best_Accuracy_Method': best_acc['Calibration'],
            'Best_Accuracy_Value': best_acc['Accuracy'],
            'Best_F1_Method': best_f1['Calibration'],
            'Best_F1_Value': best_f1['F1'],
            'Best_ECE_Method': best_ece['Calibration'],
            'Best_ECE_Value': best_ece['ECE'],
            'Best_Brier_Method': best_brier['Calibration'],
            'Best_Brier_Value': best_brier['Brier']
        })
    
    summary_df = pd.DataFrame(summary)
    summary_filepath = f'{output_dir}/calibration_best_per_model.csv'
    summary_df.to_csv(summary_filepath, index=False, float_format='%.6f')
    print(f"[SAVED] {summary_filepath}")
    
    return results_df, summary_df
