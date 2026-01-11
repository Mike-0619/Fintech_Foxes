"""
Utility script to load saved models and make predictions.

Usage:
    python load_model.py <experiment_dir> <model_type>
    
Example:
    python load_model.py fl_experiment_20260111_112536 mlp
"""

import torch
import json
import joblib
import numpy as np
from pathlib import Path
import sys

from models import CreditNet, CreditLSTM


def load_model_from_experiment(experiment_dir, model_type='mlp'):
    """
    Load a trained model from an experiment directory.
    
    Args:
        experiment_dir: Path to experiment directory
        model_type: 'mlp' or 'lstm'
        
    Returns:
        model: Loaded PyTorch model
        config: Model configuration dict
        scaler: Fitted StandardScaler
    """
    exp_path = Path(experiment_dir)
    
    # Load model configuration
    config_file = exp_path / f"{model_type}_config.json"
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")
    
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    # Create model instance
    if model_type == 'mlp':
        model = CreditNet(**config['model_kwargs'])
    elif model_type == 'lstm':
        model = CreditLSTM(**config['model_kwargs'])
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Load model weights
    weights_file = exp_path / f"{model_type}_model.pth"
    if not weights_file.exists():
        raise FileNotFoundError(f"Weights file not found: {weights_file}")
    
    model.load_state_dict(torch.load(weights_file, map_location='cpu'))
    model.eval()
    
    # Load scaler
    scaler_file = exp_path / "scaler.joblib"
    if not scaler_file.exists():
        raise FileNotFoundError(f"Scaler file not found: {scaler_file}")
    
    scaler = joblib.load(scaler_file)
    
    print(f"✅ Loaded {model_type.upper()} model from {experiment_dir}")
    print(f"   Final accuracy: {config['final_accuracy']:.4f}")
    print(f"   Hyperparameters: {config['hyperparameters']}")
    
    return model, config, scaler


def predict(model, X_raw, scaler, return_probs=False):
    """
    Make predictions on new data.
    
    Args:
        model: Trained PyTorch model
        X_raw: Raw features (before scaling) [N, 23]
        scaler: Fitted StandardScaler
        return_probs: If True, return probabilities instead of binary predictions
        
    Returns:
        predictions: Binary predictions [N] or probabilities [N]
    """
    # Scale features
    X_scaled = scaler.transform(X_raw)
    
    # Convert to tensor
    X_tensor = torch.FloatTensor(X_scaled)
    
    # Predict
    model.eval()
    with torch.no_grad():
        logits = model.main(X_tensor)
        probs = torch.sigmoid(logits).numpy().flatten()
    
    if return_probs:
        return probs
    else:
        return (probs > 0.5).astype(int)


def load_calibrated_model(experiment_dir, model_type='mlp', calibration_type='platt'):
    """
    Load a model with calibration.
    
    Args:
        experiment_dir: Path to experiment directory
        model_type: 'mlp' or 'lstm'
        calibration_type: 'platt', 'temperature', or 'beta'
        
    Returns:
        model: Loaded base model
        calibrator: Loaded calibration model
        scaler: Fitted StandardScaler
    """
    # Load base model
    model, config, scaler = load_model_from_experiment(experiment_dir, model_type)
    
    # Load calibrator
    exp_path = Path(experiment_dir)
    calib_file = exp_path / f"{calibration_type}_scaler.pth"
    
    if not calib_file.exists():
        raise FileNotFoundError(f"Calibrator file not found: {calib_file}")
    
    # Import calibrator classes
    from models import PlattScaler, TemperatureScaler, BetaCalibrator
    
    if calibration_type == 'platt':
        calibrator = PlattScaler()
    elif calibration_type == 'temperature':
        calibrator = TemperatureScaler()
    elif calibration_type == 'beta':
        calibrator = BetaCalibrator()
    else:
        raise ValueError(f"Unknown calibration type: {calibration_type}")
    
    calibrator.load_state_dict(torch.load(calib_file, map_location='cpu'))
    calibrator.eval()
    
    print(f"✅ Loaded {calibration_type} calibrator")
    
    return model, calibrator, scaler


def predict_calibrated(model, calibrator, X_raw, scaler):
    """
    Make calibrated predictions.
    
    Args:
        model: Trained base model
        calibrator: Trained calibration model
        X_raw: Raw features [N, 23]
        scaler: Fitted StandardScaler
        
    Returns:
        calibrated_probs: Calibrated probabilities [N]
    """
    # Scale features
    X_scaled = scaler.transform(X_raw)
    X_tensor = torch.FloatTensor(X_scaled)
    
    # Get logits
    model.eval()
    calibrator.eval()
    with torch.no_grad():
        logits = model.main(X_tensor)
        calibrated_probs = calibrator(logits).numpy().flatten()
    
    return calibrated_probs


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python load_model.py <experiment_dir> <model_type>")
        print("Example: python load_model.py fl_experiment_20260111_112536 mlp")
        sys.exit(1)
    
    experiment_dir = sys.argv[1]
    model_type = sys.argv[2].lower()
    
    # Load model
    model, config, scaler = load_model_from_experiment(experiment_dir, model_type)
    
    # Example: Make predictions on dummy data
    print("\n📊 Testing predictions on dummy data...")
    X_dummy = np.random.randn(5, 23)  # 5 samples, 23 features
    
    predictions = predict(model, X_dummy, scaler, return_probs=False)
    probabilities = predict(model, X_dummy, scaler, return_probs=True)
    
    print(f"\nPredictions: {predictions}")
    print(f"Probabilities: {probabilities}")
    
    print("\n✅ Model loaded successfully and ready for inference!")
