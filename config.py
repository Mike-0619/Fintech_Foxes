"""
Configuration module for Federated Learning Credit Scoring experiment.

This module centralizes all hyperparameters, constants, and configuration
settings used throughout the experiment. Modifying values here affects
the entire pipeline.

Constants are organized by category:
- Dataset Configuration
- Model Architecture
- Federated Learning Settings
- Training Hyperparameters
- Calibration Settings
- Hyperparameter Tuning

Author: FL Credit Scoring Team
"""

from typing import List, Tuple
import os
from datetime import datetime


# =============================================================================
# Dataset Configuration
# =============================================================================

DATASET_ID: int = 42477  # OpenML ID: Default of Credit Card Clients (UCI)
RANDOM_STATE: int = 42   # Global random seed for reproducibility


# =============================================================================
# Data Splitting Configuration
# =============================================================================

TEST_SIZE: float = 0.2    # 20% for final evaluation
VAL_SIZE: float = 0.3     # 30% of dev set for validation
CALIB_SIZE: float = 0.5   # 50% of val set for calibration test


# =============================================================================
# Federated Learning Configuration
# =============================================================================

NUM_CLIENTS: int = 5          # Number of simulated clients (banks)
NUM_ROUNDS: int = 15          # Number of FL communication rounds
LOCAL_EPOCHS: int = 1         # Local training epochs per round
DIRICHLET_ALPHA: float = 0.5  # Non-IID skew (lower = more heterogeneous)


# =============================================================================
# Model Architecture Configuration
# =============================================================================

INPUT_DIM: int = 23  # Total features in dataset

# LSTM-specific dimensions
LSTM_STATIC_DIM: int = 5    # Static: LIMIT_BAL, SEX, EDUCATION, MARRIAGE, AGE
LSTM_TEMPORAL_DIM: int = 1  # Deprecated: now uses 3 features per timestep
LSTM_HIDDEN: int = 64       # LSTM hidden size
LSTM_LAYERS: int = 4        # Number of LSTM layers


# =============================================================================
# Training Hyperparameters
# =============================================================================

LEARNING_RATE: float = 0.005
BATCH_SIZE: int = 64
DROPOUT: float = 0.2


# =============================================================================
# Feature Names (UCI Dataset Schema)
# =============================================================================

UCI_FEATURE_NAMES: List[str] = [
    'LIMIT_BAL',    # x1:  Credit limit
    'SEX',          # x2:  Gender (1=male, 2=female) - PROTECTED ATTRIBUTE
    'EDUCATION',    # x3:  Education level
    'MARRIAGE',     # x4:  Marital status
    'AGE',          # x5:  Age in years
    'PAY_0',        # x6:  Repayment status (Sep 2005)
    'PAY_2',        # x7:  Repayment status (Aug 2005)
    'PAY_3',        # x8:  Repayment status (Jul 2005)
    'PAY_4',        # x9:  Repayment status (Jun 2005)
    'PAY_5',        # x10: Repayment status (May 2005)
    'PAY_6',        # x11: Repayment status (Apr 2005)
    'BILL_AMT1',    # x12: Bill statement (Sep 2005)
    'BILL_AMT2',    # x13: Bill statement (Aug 2005)
    'BILL_AMT3',    # x14: Bill statement (Jul 2005)
    'BILL_AMT4',    # x15: Bill statement (Jun 2005)
    'BILL_AMT5',    # x16: Bill statement (May 2005)
    'BILL_AMT6',    # x17: Bill statement (Apr 2005)
    'PAY_AMT1',     # x18: Payment amount (Sep 2005)
    'PAY_AMT2',     # x19: Payment amount (Aug 2005)
    'PAY_AMT3',     # x20: Payment amount (Jul 2005)
    'PAY_AMT4',     # x21: Payment amount (Jun 2005)
    'PAY_AMT5',     # x22: Payment amount (May 2005)
    'PAY_AMT6'      # x23: Payment amount (Apr 2005)
]

# Feature indices for LSTM architecture
STATIC_FEATURE_INDICES: List[int] = [0, 1, 2, 3, 4]  # First 5 features
TEMPORAL_FEATURE_INDICES: List[int] = [5, 6, 7, 8, 9, 10]  # PAY_0 to PAY_6


# =============================================================================
# Calibration Configuration
# =============================================================================

ECE_N_BINS: int = 10       # Bins for Expected Calibration Error
PLATT_LR: float = 0.01     # Learning rate for Platt/Beta calibration
PLATT_MAX_ITER: int = 100  # Max iterations for calibration


# =============================================================================
# Hyperparameter Tuning Configuration
# =============================================================================

TUNE_N_TRIALS: int = 10    # Number of tuning trials
TUNE_PATIENCE: int = 5     # Early stopping patience
TUNE_MIN_DELTA: float = 0.001  # Minimum improvement threshold
TUNE_MAX_EPOCHS: int = 20  # Max epochs during tuning


# =============================================================================
# Fairness Configuration
# =============================================================================

PROTECTED_ATTR_INDEX: int = 1       # SEX attribute index
PROTECTED_ATTR_NAME: str = "SEX"    # Name for reporting
FAIRNESS_THRESHOLD: float = 0.05   # 5% gap threshold


# =============================================================================
# Utility Functions
# =============================================================================

def get_output_dir() -> str:
    """
    Generate timestamped output directory for experiment results.
    
    Returns:
        Path to output directory (created if not exists)
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"fl_experiment_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    return output_dir
