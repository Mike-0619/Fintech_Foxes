"""
Fairness auditing module for federated credit scoring.

Implements fairness metrics and auditing functions for evaluating
model performance across protected groups.

Metrics Implemented:
- Demographic Parity (Statistical Parity)
- Equalized Odds
- Equal Opportunity

Reference: Hardt et al. (2016) "Equality of Opportunity in Supervised Learning"
"""

from typing import Dict, Tuple, Optional
import numpy as np
from sklearn.metrics import confusion_matrix

# Protected attribute configuration
PROTECTED_ATTR_INDEX = 1  # SEX: index 1 in feature array (after scaling)
PROTECTED_ATTR_NAME = "SEX"


def split_by_protected_attribute(
    X: np.ndarray,
    y: np.ndarray,
    y_pred: np.ndarray,
    protected_idx: int = PROTECTED_ATTR_INDEX
) -> Tuple[Dict, Dict]:
    """
    Split data by protected attribute into groups.
    
    Args:
        X: Feature matrix [N, D]
        y: True labels [N]
        y_pred: Predicted labels or probabilities [N]
        protected_idx: Index of protected attribute in X
        
    Returns:
        groups: Dict mapping group_id -> indices
        stats: Dict with group statistics
    """
    protected_values = X[:, protected_idx]
    
    # Use median split for continuous (scaled) values
    # FIX: Use strict < to ensure both groups have samples
    median_val = np.median(protected_values)
    group_0_mask = protected_values < median_val
    group_1_mask = protected_values >= median_val
    
    # If all values are the same (median split fails), use percentile split
    if group_0_mask.sum() == 0 or group_1_mask.sum() == 0:
        # Use 50th percentile with strict split
        percentile_50 = np.percentile(protected_values, 50)
        # Split by unique values if available
        unique_vals = np.unique(protected_values)
        if len(unique_vals) >= 2:
            # Use the midpoint between two middle values
            mid_idx = len(unique_vals) // 2
            threshold = (unique_vals[mid_idx-1] + unique_vals[mid_idx]) / 2
            group_0_mask = protected_values < threshold
            group_1_mask = protected_values >= threshold
        else:
            # Fallback: split randomly 50/50
            n = len(protected_values)
            indices = np.random.permutation(n)
            group_0_mask = np.zeros(n, dtype=bool)
            group_1_mask = np.zeros(n, dtype=bool)
            group_0_mask[indices[:n//2]] = True
            group_1_mask[indices[n//2:]] = True
    
    groups = {
        0: np.where(group_0_mask)[0],
        1: np.where(group_1_mask)[0]
    }
    
    stats = {
        'group_0_size': len(groups[0]),
        'group_1_size': len(groups[1]),
        'group_0_base_rate': y[groups[0]].mean() if len(groups[0]) > 0 else 0,
        'group_1_base_rate': y[groups[1]].mean() if len(groups[1]) > 0 else 0
    }
    
    return groups, stats


def demographic_parity(
    y_pred: np.ndarray,
    groups: Dict[int, np.ndarray]
) -> Dict[str, float]:
    """
    Compute Demographic Parity (Statistical Parity) metrics.
    
    Demographic Parity requires: P(Ŷ=1|A=0) = P(Ŷ=1|A=1)
    
    Args:
        y_pred: Predicted labels (binary) [N]
        groups: Dict mapping group_id -> indices
        
    Returns:
        Dict with positive rates per group and DP difference
    """
    # Convert probabilities to binary if needed
    if y_pred.dtype == float and y_pred.max() <= 1:
        y_pred_binary = (y_pred > 0.5).astype(int)
    else:
        y_pred_binary = y_pred.astype(int)
    
    # Positive prediction rate per group
    rate_0 = y_pred_binary[groups[0]].mean() if len(groups[0]) > 0 else 0
    rate_1 = y_pred_binary[groups[1]].mean() if len(groups[1]) > 0 else 0
    
    dp_diff = abs(rate_0 - rate_1)
    dp_ratio = min(rate_0, rate_1) / max(rate_0, rate_1) if max(rate_0, rate_1) > 0 else 1.0
    
    return {
        'group_0_positive_rate': rate_0,
        'group_1_positive_rate': rate_1,
        'demographic_parity_diff': dp_diff,
        'demographic_parity_ratio': dp_ratio,
        'satisfies_dp': dp_diff < 0.05  # 5% threshold
    }


def equalized_odds(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: Dict[int, np.ndarray]
) -> Dict[str, float]:
    """
    Compute Equalized Odds metrics.
    
    Equalized Odds requires:
    - P(Ŷ=1|Y=1,A=0) = P(Ŷ=1|Y=1,A=1)  (Equal TPR)
    - P(Ŷ=1|Y=0,A=0) = P(Ŷ=1|Y=0,A=1)  (Equal FPR)
    
    Args:
        y_true: True labels [N]
        y_pred: Predicted labels (binary) [N]
        groups: Dict mapping group_id -> indices
        
    Returns:
        Dict with TPR, FPR per group and EO differences
    """
    # Convert probabilities to binary if needed
    if y_pred.dtype == float and y_pred.max() <= 1:
        y_pred_binary = (y_pred > 0.5).astype(int)
    else:
        y_pred_binary = y_pred.astype(int)
    
    results = {}
    
    for group_id, indices in groups.items():
        if len(indices) == 0:
            results[f'group_{group_id}_tpr'] = 0
            results[f'group_{group_id}_fpr'] = 0
            continue
            
        y_g = y_true[indices]
        y_pred_g = y_pred_binary[indices]
        
        # True Positive Rate: P(Ŷ=1|Y=1)
        positives = y_g == 1
        if positives.sum() > 0:
            tpr = y_pred_g[positives].mean()
        else:
            tpr = 0
            
        # False Positive Rate: P(Ŷ=1|Y=0)
        negatives = y_g == 0
        if negatives.sum() > 0:
            fpr = y_pred_g[negatives].mean()
        else:
            fpr = 0
            
        results[f'group_{group_id}_tpr'] = tpr
        results[f'group_{group_id}_fpr'] = fpr
    
    # Compute differences
    tpr_diff = abs(results['group_0_tpr'] - results['group_1_tpr'])
    fpr_diff = abs(results['group_0_fpr'] - results['group_1_fpr'])
    
    results['tpr_diff'] = tpr_diff
    results['fpr_diff'] = fpr_diff
    results['equalized_odds_diff'] = tpr_diff + fpr_diff
    results['satisfies_eo'] = (tpr_diff < 0.05) and (fpr_diff < 0.05)
    
    return results


def equal_opportunity(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: Dict[int, np.ndarray]
) -> Dict[str, float]:
    """
    Compute Equal Opportunity metrics.
    
    Equal Opportunity requires: P(Ŷ=1|Y=1,A=0) = P(Ŷ=1|Y=1,A=1)
    (Only TPR equality, not FPR)
    
    Args:
        y_true: True labels [N]
        y_pred: Predicted labels [N]
        groups: Dict mapping group_id -> indices
        
    Returns:
        Dict with TPR per group and difference
    """
    eo_results = equalized_odds(y_true, y_pred, groups)
    
    return {
        'group_0_tpr': eo_results['group_0_tpr'],
        'group_1_tpr': eo_results['group_1_tpr'],
        'equal_opportunity_diff': eo_results['tpr_diff'],
        'satisfies_eop': eo_results['tpr_diff'] < 0.05
    }


def compute_fairness_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    X: np.ndarray,
    protected_idx: int = PROTECTED_ATTR_INDEX
) -> Dict[str, float]:
    """
    Compute all fairness metrics.
    
    Args:
        y_true: True labels [N]
        y_pred: Predicted labels or probabilities [N]
        X: Feature matrix [N, D]
        protected_idx: Index of protected attribute
        
    Returns:
        Combined fairness metrics dictionary
    """
    groups, group_stats = split_by_protected_attribute(X, y_true, y_pred, protected_idx)
    
    dp_metrics = demographic_parity(y_pred, groups)
    eo_metrics = equalized_odds(y_true, y_pred, groups)
    eop_metrics = equal_opportunity(y_true, y_pred, groups)
    
    return {
        **group_stats,
        **dp_metrics,
        **eo_metrics,
        **eop_metrics
    }


def fairness_report(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    device: str = 'cpu',
    protected_idx: int = PROTECTED_ATTR_INDEX,
    protected_name: str = PROTECTED_ATTR_NAME
) -> Dict[str, float]:
    """
    Generate comprehensive fairness audit report for a model.
    
    Args:
        model: Trained PyTorch model
        X_test: Test features [N, D]
        y_test: Test labels [N]
        device: PyTorch device
        protected_idx: Index of protected attribute
        protected_name: Name of protected attribute
        
    Returns:
        Dict with all fairness metrics
    """
    import torch
    
    model.eval()
    with torch.no_grad():
        X_tensor = torch.Tensor(X_test).to(device)
        # Get raw logits from model.main() to avoid double sigmoid
        logits = model.main(X_tensor)
        y_pred_probs = torch.sigmoid(logits).cpu().numpy().flatten()
    
    metrics = compute_fairness_metrics(y_test, y_pred_probs, X_test, protected_idx)
    
    # Print report
    print("\n" + "=" * 60)
    print(f"FAIRNESS AUDIT REPORT (Protected: {protected_name})")
    print("=" * 60)
    print(f"\nGroup Statistics:")
    print(f"  Group 0 size: {metrics['group_0_size']}")
    print(f"  Group 1 size: {metrics['group_1_size']}")
    print(f"  Group 0 base rate: {metrics['group_0_base_rate']:.4f}")
    print(f"  Group 1 base rate: {metrics['group_1_base_rate']:.4f}")
    
    print(f"\nDemographic Parity:")
    print(f"  Group 0 positive rate: {metrics['group_0_positive_rate']:.4f}")
    print(f"  Group 1 positive rate: {metrics['group_1_positive_rate']:.4f}")
    print(f"  DP Difference: {metrics['demographic_parity_diff']:.4f} {'✓' if metrics['satisfies_dp'] else '✗'}")
    
    print(f"\nEqualized Odds:")
    print(f"  Group 0 TPR: {metrics['group_0_tpr']:.4f}, FPR: {metrics['group_0_fpr']:.4f}")
    print(f"  Group 1 TPR: {metrics['group_1_tpr']:.4f}, FPR: {metrics['group_1_fpr']:.4f}")
    print(f"  TPR Difference: {metrics['tpr_diff']:.4f}")
    print(f"  FPR Difference: {metrics['fpr_diff']:.4f}")
    print(f"  EO Difference: {metrics['equalized_odds_diff']:.4f} {'✓' if metrics['satisfies_eo'] else '✗'}")
    
    print(f"\nEqual Opportunity:")
    print(f"  Difference: {metrics['equal_opportunity_diff']:.4f} {'✓' if metrics['satisfies_eop'] else '✗'}")
    
    print("=" * 60)
    
    return metrics


def plot_fairness_comparison(
    model_metrics: Dict[str, Dict],
    output_dir: str
) -> None:
    """
    Plot fairness metrics comparison across models.
    
    Args:
        model_metrics: Dict mapping model_name -> fairness_metrics
        output_dir: Directory to save plot
    """
    import matplotlib.pyplot as plt
    
    models = list(model_metrics.keys())
    dp_diffs = [model_metrics[m]['demographic_parity_diff'] for m in models]
    eo_diffs = [model_metrics[m]['equalized_odds_diff'] for m in models]
    eop_diffs = [model_metrics[m]['equal_opportunity_diff'] for m in models]
    
    x = np.arange(len(models))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    bars1 = ax.bar(x - width, dp_diffs, width, label='Demographic Parity', alpha=0.8)
    bars2 = ax.bar(x, eo_diffs, width, label='Equalized Odds', alpha=0.8)
    bars3 = ax.bar(x + width, eop_diffs, width, label='Equal Opportunity', alpha=0.8)
    
    # Add threshold line
    ax.axhline(y=0.05, color='red', linestyle='--', label='5% Threshold')
    
    ax.set_ylabel('Fairness Gap')
    ax.set_title('Fairness Metrics Comparison (Lower is Better)')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/fairness_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"[SAVED] {output_dir}/fairness_comparison.png")
