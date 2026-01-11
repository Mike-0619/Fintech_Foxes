# 🏦 Federated Learning for Credit Scoring

> Privacy-preserving credit risk assessment using federated learning - achieving 81% accuracy without sharing customer data

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 🎯 Problem

Banks need accurate credit scoring models but cannot share customer data due to privacy laws (GDPR, CCPA). Local models trained on individual bank data perform poorly (56% accuracy), while centralized models (if legal) achieve 81% accuracy.

**Solution:** Federated Learning trains a shared model without exchanging raw data - only model weights.

---

## ✨ Key Results

| Approach | Accuracy | Privacy | Data Sharing |
|----------|----------|---------|--------------|
| **Local Only** | 56.6% | ✅ Full | None |
| **FL-MLP** | 79.4% | ✅ Full | Weights only |
| **FL-LSTM** | 81.0% | ✅ Full | Weights only |
| **Centralized** | 81.3% | ❌ None | All data |

**Key Finding:** FL achieves 99.7% of centralized performance with zero data sharing.

---

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/your-org/federated-credit-scoring.git
cd federated-credit-scoring/fl_credit_scoring

# Install dependencies
pip install -r requirements.txt
```

### Run Experiment

```bash
# Basic run (~5 min)
python main.py --no-shap --no-lime

# Full analysis (~15 min)
python main.py

# With hyperparameter tuning (~45 min)
python main.py --tune --tune-trials 10
```

### Command Options

```bash
python main.py [OPTIONS]

--tune              Enable hyperparameter optimization
--tune-trials N     Number of optimization trials (default: 10)
--no-shap          Skip SHAP explainability analysis
--no-lime          Skip LIME explainability analysis
```

---

## 📁 Project Structure

```
fl_credit_scoring/
│
├── main.py                 # Orchestrator - runs full experiment
├── config.py              # Hyperparameters & constants
├── data.py                # Data loading & preprocessing
├── models.py              # Neural architectures (MLP, LSTM)
├── federated.py           # FedAvg implementation
├── calibration.py         # 4 calibration methods
├── baselines.py           # Comparison models (LogReg, XGBoost)
├── hyperparams.py         # Optuna optimization
├── visualization.py       # Plotting suite
└── requirements.txt       # Dependencies

Output (fl_experiment_*/):
├── mlp_model.pth          # Trained FL-MLP
├── lstm_model.pth         # Trained FL-LSTM
├── final_model_comparison.csv
├── calibration_results.csv
└── Fig1_Convergence.png
```

---

## 🛠️ Core Components

### 1. Data Pipeline

**Dataset:** UCI Credit Card Default (30K samples, 23 features)

**Splits:**
- Train: 16,800 samples (federated across 5 clients)
- Calibration: 3,600 samples
- Evaluation: 3,600 samples  
- Test: 6,000 samples

**Non-IID Simulation:** Dirichlet(α=0.5) creates heterogeneous data distributions

### 2. Model Architectures

**FL-MLP:**
```
Input(23) → Dense(256) → Dense(96) → Dense(1) → Sigmoid
```

**FL-LSTM:**
```
Static(5) → Dense(32) ─┐
Temporal(18) → LSTM(32, 4 layers) ─┤→ Concat → Dense(64) → Dense(32) → Dense(1)
```

### 3. Federated Training

**Algorithm:** FedAvg (McMahan et al., 2017)

```
FOR round = 1 to 15:
  1. Server sends global model to clients
  2. Each client trains on local data
  3. Clients send updated weights to server
  4. Server aggregates: w = Σ (n_k/N) × w_k
```

**Key:** Only weights are shared, never raw customer data.

### 4. Calibration Methods

| Method | Parameters | Best For |
|--------|-----------|----------|
| Platt | A, B | Binary classification |
| Temperature | T | Multi-class (preserves accuracy) |
| Beta | a, b, c | Complex miscalibration |
| FedCal | Aggregated A, B | Federated setting |

---

## 📊 Results Summary

### Model Performance

```
FL-MLP:   79.4% accuracy, ECE=0.056 (Beta calibrated)
FL-LSTM:  81.0% accuracy, ECE=0.072 (Platt calibrated)
Central:  81.3% accuracy, ECE=0.030 (Temperature calibrated)
```

### Calibration Impact

```
Before: ECE = 0.227 (miscalibrated)
After:  ECE = 0.056 (well-calibrated)
Improvement: 75.3% reduction
```

### Convergence

Both models converge within 15 rounds:
- Round 1: 73-76%
- Round 5: 73-77%
- Round 15: 73-78% (stable)

---

## 🔧 Configuration

Edit `config.py` to customize:

```python
# Federated Learning
NUM_CLIENTS = 5           # Number of banks
NUM_ROUNDS = 15           # FL rounds
DIRICHLET_ALPHA = 0.5     # Non-IID intensity

# Model Architecture
LEARNING_RATE = 0.005
BATCH_SIZE = 64
DROPOUT = 0.2

# Optimization
TUNE_N_TRIALS = 10        # Optuna trials
```

---

## 📦 Output Files

Each run creates `fl_experiment_YYYYMMDD_HHMMSS/` with:

**Models:**
- `mlp_model.pth` - FL-MLP weights
- `lstm_model.pth` - FL-LSTM weights
- `*_scaler.pth` - Calibrators

**Results:**
- `final_model_comparison.csv` - Performance table
- `calibration_detailed_results.csv` - All calibration metrics
- `best_hyperparams.json` - Optimal hyperparameters

**Visualizations:**
- `Fig1_Convergence.png` - Training curves
- `Fig2_Calibration.png` - Reliability diagrams
- `Fig3_FinalComparison.png` - Model comparison

---

## 🔍 Model Inference

```python
from load_model import load_model_from_experiment

# Load trained model
model, config, scaler = load_model_from_experiment(
    'fl_experiment_20260111_133448',
    model_type='mlp'
)

# Make predictions
import torch
X_new = scaler.transform(X_raw)
predictions = model(torch.Tensor(X_new))
```

---

## 📚 Key Dependencies

```
torch>=2.0.0              # Deep learning
scikit-learn>=1.3.0       # Preprocessing & baselines
optuna>=3.3.0             # Hyperparameter optimization
xgboost>=2.0.0            # Gradient boosting baseline
shap>=0.42.0              # Model explainability
```

Install all: `pip install -r requirements.txt`

---

## 🎓 Citation

```bibtex
@software{federated_credit_scoring_2026,
  author = {Your Name},
  title = {Federated Learning for Credit Scoring},
  year = {2026},
  url = {https://github.com/your-org/federated-credit-scoring}
}
```

**Related Papers:**
- McMahan et al. (2017). "Communication-Efficient Learning of Deep Networks from Decentralized Data." AISTATS
- Guo et al. (2017). "On Calibration of Modern Neural Networks." ICML

---

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

---

## 🤝 Contributing

Issues and pull requests welcome! Please:
1. Fork the repository
2. Create feature branch (`git checkout -b feature/name`)
3. Commit changes (`git commit -am 'Add feature'`)
4. Push to branch (`git push origin feature/name`)
5. Open Pull Request

---

## 📞 Contact

- **Issues:** [GitHub Issues](https://github.com/your-org/federated-credit-scoring/issues)
- **Email:** support@your-org.com

---

**⭐ Star this repo if it helps your research!**
- **`shap_summary_mlp.png`**: Feature importance analysis.

## Key Metrics
- **Accuracy**: Overall correctness of predictions.
- **F1-Score**: Harmonic mean of precision and recall, crucial for imbalanced credit default data.
- **ECE (Expected Calibration Error)**: Measures the differene between predicted probabilities and actual outcomes. Lower is better.
- **Brier Score**: Measures the mean squared difference between predicted probability and actual outcome. Lower is better.
