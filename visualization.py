"""
Polished visualization module for Federated Learning Credit Scoring.
Produces publication-quality figures:
1. Fig1_Convergence: FL training progress
2. Fig2_Calibration: Reliability diagrams for all models
3. Fig3_FinalComparison: Performance metrics across all models
4. LIME Explanations
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import torch
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('deep')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['figure.dpi'] = 300

def plot_convergence(mlp_accs, lstm_accs, output_dir):
    """
    Plot training convergence for FL models side-by-side.
    """
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # MLP
        ax1.plot(range(1, len(mlp_accs) + 1), mlp_accs, 'o-', color='#1f77b4', linewidth=2)
        ax1.set_title("FL-FedAvg (MLP)", fontsize=14, fontweight='bold')
        ax1.set_xlabel("Round")
        ax1.set_ylabel("Accuracy")
        ax1.set_ylim([0.4, 0.9])
        
        # LSTM
        ax2.plot(range(1, len(lstm_accs) + 1), lstm_accs, 's-', color='#ff7f0e', linewidth=2)
        ax2.set_title("FL-FedAvg (LSTM)", fontsize=14, fontweight='bold')
        ax2.set_xlabel("Round")
        ax2.set_ylim([0.4, 0.9])
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/Fig1_Convergence.png", bbox_inches='tight')
        plt.close()
        print(f"[SAVED] {output_dir}/Fig1_Convergence.png")
    except Exception as e:
        print(f"Error plotting convergence: {e}")

def plot_reliability_curve(y_true, probs, ax, label, color):
    """
    Helper to plot a single reliability curve.
    """
    fraction_of_positives, mean_predicted_value = calibration_curve(y_true, probs, n_bins=10)
    brier = brier_score_loss(y_true, probs)
    ax.plot(mean_predicted_value, fraction_of_positives, "s-", 
            label=f"{label} (Brier={brier:.3f})", color=color, linewidth=2)

def plot_calibration_summary(all_models, calibrators_dict, X_eval, y_eval, device, output_dir):
    """
    Plot reliability diagrams for MLP, LSTM, and Central NN.
    Compares Uncalibrated vs Best Calibrated version.
    """
    try:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        model_names = ['FL-FedAvg (MLP)', 'FL-FedAvg (LSTM)', 'Central NN']
        
        # Map names if they differ in all_models keys
        # Assuming keys are 'FL-FedAvg (MLP)', 'FL-FedAvg (LSTM)', 'Central Neural Network' (or 'Central NN'?)
        # Let's handle generic keys
        from difflib import get_close_matches
        
        for i, model_target_name in enumerate(model_names):
            ax = axes[i]
            
            # Find actual key in all_models
            keys = list(all_models.keys())
            matches = get_close_matches(model_target_name, keys, n=1, cutoff=0.5)
            if not matches:
                continue
            model_key = matches[0]
            
            # Plot perfect calibration line
            ax.plot([0, 1], [0, 1], "k:", label="Perfect", alpha=0.6)
            
            model = all_models[model_key]
            model.eval()
            with torch.no_grad():
                logits = model.main(torch.Tensor(X_eval).to(device)).cpu()
            
            # 1. Uncalibrated
            probs = torch.sigmoid(logits).numpy().flatten()
            plot_reliability_curve(y_eval, probs, ax, "Uncalibrated", '#95a5a6') # Grey
            
            # 2. Best Calibrated (Use Beta/Platt if available)
            # Try to grab 'Beta' or 'Platt' for this model
            calib_key = None
            if (model_key, 'Beta') in calibrators_dict:
                calib_key = (model_key, 'Beta')
                label = "Beta"
            elif (model_key, 'Platt') in calibrators_dict:
                calib_key = (model_key, 'Platt')
                label = "Platt"
                
            if calib_key:
                calibrator = calibrators_dict[calib_key]
                calibrator.eval()
                with torch.no_grad():
                    calib_probs = calibrator(logits.to(device)).cpu().numpy().flatten()
                plot_reliability_curve(y_eval, calib_probs, ax, f"{label} Calib", '#2ecc71') # Green
            
            ax.set_title(model_key, fontsize=14, fontweight='bold')
            ax.set_xlabel("Mean Predicted Probability")
            if i == 0:
                ax.set_ylabel("Fraction of Positives")
            ax.set_ylim([0, 1])
            ax.set_xlim([0, 1])
            ax.legend(loc="lower right", fontsize=9)
            ax.grid(True, alpha=0.3)
            
        plt.tight_layout()
        plt.savefig(f"{output_dir}/Fig2_Calibration.png", bbox_inches='tight')
        plt.close()
        print(f"[SAVED] {output_dir}/Fig2_Calibration.png")
    except Exception as e:
        print(f"Error plotting calibration summary: {e}")


def plot_final_comparison(results_summary, output_dir):
    """
    Bar chart comparing Accuracy, F1, and ECE for all models.
    """
    try:
        # Convert dict to DataFrame if needed
        if isinstance(results_summary, dict):
            df = pd.DataFrame(results_summary).T.reset_index()
            df.columns = ['Model', 'Accuracy', 'AUC', 'F1', 'ECE']
        else:
            df = results_summary

        # Clean names
        df['Model'] = df['Model'].replace({
            'FL-FedAvg (MLP)': 'FL-MLP',
            'FL-FedAvg (LSTM)': 'FL-LSTM',
            'Central Neural Network': 'Central NN'
        })

        # Melt for plotting
        df_melt = df.melt(id_vars='Model', value_vars=['Accuracy', 'F1', 'ECE'], 
                          var_name='Metric', value_name='Score')
        
        plt.figure(figsize=(10, 6))
        sns.barplot(data=df_melt, x='Model', y='Score', hue='Metric', palette=['#3498db', '#e74c3c', '#2ecc71'])
        
        plt.title("Final Model Comparison (Calibrated)", fontsize=16, fontweight='bold')
        plt.xlabel("")
        plt.ylim([0, 1.0])
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/Fig3_FinalComparison.png", bbox_inches='tight')
        plt.close()
        print(f"[SAVED] {output_dir}/Fig3_FinalComparison.png")
    except Exception as e:
        print(f"Error plotting comparison: {e}")

def generate_lime_explanation(model, X_train, X_test, y_test, feature_names, sample_idx, output_dir):
    """
    Generate LIME explanation for a single instance.
    """
    try:
        import lime
        import lime.lime_tabular
        
        # Define prediction function for LIME (needs numpy input, returns probas)
        def predict_fn(x_numpy):
            model.eval()
            tensor_x = torch.Tensor(x_numpy)
            device = next(model.parameters()).device
            tensor_x = tensor_x.to(device)
            with torch.no_grad():
                logits = model.main(tensor_x)
                probs = torch.sigmoid(logits).cpu().numpy()
            # LIME expects [prob_class_0, prob_class_1]
            return np.hstack([1-probs, probs])

        explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=X_train,
            feature_names=feature_names,
            class_names=['Non-Default', 'Default'],
            mode='classification'
        )
        
        # Explain the instance
        instance = X_test[sample_idx]
        exp = explainer.explain_instance(
            data_row=instance, 
            predict_fn=predict_fn, 
            num_features=10
        )
        
        # Save results
        exp.save_to_file(f"{output_dir}/lime_explanation.html")
        fig = exp.as_pyplot_figure()
        plt.tight_layout()
        plt.savefig(f"{output_dir}/lime_explanation.png")
        plt.close()
        print(f"[SAVED] {output_dir}/lime_explanation.png")
        
    except ImportError:
        print("LIME not installed via pip.")
    except Exception as e:
        print(f"LIME generation failed: {e}")
