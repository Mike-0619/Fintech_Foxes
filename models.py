"""
Model definitions: CreditNet (MLP), CreditLSTM, and Calibration Scalers
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import INPUT_DIM, LSTM_STATIC_DIM, LSTM_TEMPORAL_DIM, LSTM_HIDDEN, LSTM_LAYERS


class CreditNet(nn.Module):
    """
    Standard MLP for credit scoring.

    Architecture is configurable via hyperparameters:
        Default: Dense(64) -> Dense(32) -> Output(1)

    Args:
        input_dim: Number of input features
        hidden_layers: List of hidden layer sizes (default [64, 32])
        dropout: Dropout rate (default 0.2)
    """

    def __init__(self, input_dim=INPUT_DIM, hidden_layers=None, dropout=0.2):
        super(CreditNet, self).__init__()

        if hidden_layers is None:
            hidden_layers = [64, 32]

        # Build sequential layers dynamically
        layers = []
        prev_dim = input_dim
        for i, h in enumerate(hidden_layers):
            layers.append(nn.Linear(prev_dim, h))
            # Add BatchNorm and ReLU for all but last hidden layer
            if i < len(hidden_layers) - 1:
                layers.append(nn.BatchNorm1d(h))
                layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 1))

        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return torch.sigmoid(self.main(x))


class CreditLSTM(nn.Module):
    """
    LSTM-based credit scoring model using ALL 23 features.
    
    FIXED: Now uses all features instead of discarding 12.

    Hybrid architecture:
    - Static features (5): LIMIT_BAL, SEX, EDUCATION, MARRIAGE, AGE
    - Temporal payment status (6): PAY_0, PAY_2, PAY_3, PAY_4, PAY_5, PAY_6
    - Temporal bill amounts (6): BILL_AMT1-6
    - Temporal payment amounts (6): PAY_AMT1-6
    
    The 18 temporal features are reshaped into a sequence of 6 timesteps x 3 features:
    [PAY_status, BILL_AMT, PAY_AMT] for each of 6 months.

    Args:
        static_dim: Number of static features (default 5)
        temporal_features_per_step: Features per timestep (default 3: pay_status, bill, pay_amt)
        seq_length: Number of timesteps (default 6: months)
        lstm_hidden: LSTM hidden size (default 64)
        lstm_layers: Number of LSTM layers (default 4)
        dropout: Dropout rate (default 0.2)
    """

    def __init__(self, static_dim=5, temporal_features_per_step=3, seq_length=6,
                 lstm_hidden=LSTM_HIDDEN, lstm_layers=LSTM_LAYERS, dropout=0.2):
        super(CreditLSTM, self).__init__()
        
        self.seq_length = seq_length
        self.temporal_features_per_step = temporal_features_per_step

        # Static feature branch (5 features)
        self.static_fc = nn.Sequential(
            nn.Linear(static_dim, 32),
            nn.LayerNorm(32),  # LayerNorm instead of BatchNorm for FL stability
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Temporal feature branch (LSTM)
        # Input: [batch, 6 timesteps, 3 features per step]
        self.lstm = nn.LSTM(
            input_size=temporal_features_per_step,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            dropout=dropout if lstm_layers > 1 else 0,  # LSTM dropout only works with >1 layer
            batch_first=True
        )

        # Combined layers
        combined_dim = 32 + lstm_hidden
        self.combined = nn.Sequential(
            nn.Linear(combined_dim, 64),
            nn.LayerNorm(64),  # LayerNorm instead of BatchNorm for FL stability
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.LayerNorm(32),  # LayerNorm instead of BatchNorm for FL stability
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
        # Initialize weights properly
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights to prevent getting stuck in local minima."""
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() >= 2:  # Skip 1D weights (BatchNorm)
                if 'lstm' in name:
                    # Xavier initialization for LSTM
                    nn.init.xavier_uniform_(param)
                else:
                    # He initialization for ReLU layers
                    nn.init.kaiming_normal_(param, mode='fan_out', nonlinearity='relu')
            elif 'bias' in name:
                nn.init.constant_(param, 0)

    def main(self, x):
        """Return logits (before sigmoid) for calibration."""
        # Feature indices based on UCI dataset:
        # 0-4: Static (LIMIT_BAL, SEX, EDUCATION, MARRIAGE, AGE)
        # 5-10: PAY_0, PAY_2, PAY_3, PAY_4, PAY_5, PAY_6 (payment status)
        # 11-16: BILL_AMT1-6
        # 17-22: PAY_AMT1-6
        
        # Extract static features
        static = x[:, :5]  # [batch, 5]
        
        # Extract and reshape temporal features for LSTM
        # Create sequence: [batch, 6 timesteps, 3 features]
        # Each timestep: [pay_status, bill_amt, pay_amt]
        pay_status = x[:, 5:11]    # [batch, 6] - payment status per month
        bill_amts = x[:, 11:17]    # [batch, 6] - bill amounts per month
        pay_amts = x[:, 17:23]     # [batch, 6] - payment amounts per month
        
        # Stack to create [batch, 6, 3] tensor
        # Each of the 6 timesteps has 3 features: [pay_status, bill_amt, pay_amt]
        temporal = torch.stack([pay_status, bill_amts, pay_amts], dim=2)  # [batch, 6, 3]

        # Process static features
        static_out = self.static_fc(static)

        # Process temporal features with LSTM
        # Initialize hidden state explicitly for clarity
        batch_size = x.size(0)
        h0 = torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size, device=x.device)
        c0 = torch.zeros(self.lstm.num_layers, batch_size, self.lstm.hidden_size, device=x.device)
        
        lstm_out, _ = self.lstm(temporal, (h0, c0))
        lstm_out = lstm_out[:, -1, :]  # Take last timestep [batch, hidden]

        # Combine and predict
        combined = torch.cat([static_out, lstm_out], dim=1)
        logits = self.combined(combined)
        return logits

    def forward(self, x):
        return torch.sigmoid(self.main(x))


# ============================================================================
# CALIBRATION SCALERS
# ============================================================================

class PlattScaler(nn.Module):
    """
    Platt Scaling for binary calibration.

    Learns parameters A and B: P_calibrated = sigmoid(A * logits + B)
    """

    def __init__(self):
        super(PlattScaler, self).__init__()
        self.A = nn.Parameter(torch.ones(1))
        self.B = nn.Parameter(torch.zeros(1))

    def forward(self, logits):
        return torch.sigmoid(self.A * logits + self.B)

    def fit(self, logits, labels, lr=0.01, max_iter=100):
        optimizer = torch.optim.LBFGS([self.A, self.B], lr=lr, max_iter=max_iter)

        def closure():
            optimizer.zero_grad()
            out = self.forward(logits)
            loss = F.binary_cross_entropy(out, labels)
            loss.backward()
            return loss

        optimizer.step(closure)
        print(f"  Platt Parameters: A={self.A.item():.4f}, B={self.B.item():.4f}")


class TemperatureScaler(nn.Module):
    """
    Temperature Scaling for binary calibration.

    Learns a single temperature parameter T: P_calibrated = sigmoid(logits / T)

    When T > 1: Softens predictions (pushes toward 0.5)
    When T < 1: Sharpens predictions (pushes toward 0 or 1)
    """

    def __init__(self):
        super(TemperatureScaler, self).__init__()
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, logits):
        return torch.sigmoid(logits / self.temperature)

    def fit(self, logits, labels, lr=0.01, max_iter=50):
        optimizer = torch.optim.LBFGS([self.temperature], lr=lr, max_iter=max_iter)

        def closure():
            optimizer.zero_grad()
            out = self.forward(logits)
            loss = F.binary_cross_entropy(out, labels)
            loss.backward()
            return loss

        optimizer.step(closure)
        print(f"  Learned Temperature: T = {self.temperature.item():.4f}")


class FedCalPlattScaler(nn.Module):
    """
    Federated Calibration Scaler (FedCal, Peng et al., 2024).

    Used for per-client calibration in FedCal algorithm.
    """

    def __init__(self):
        super(FedCalPlattScaler, self).__init__()
        self.A = nn.Parameter(torch.ones(1))
        self.B = nn.Parameter(torch.zeros(1))

    def forward(self, logits):
        return torch.sigmoid(self.A * logits + self.B)

    def fit(self, logits, labels, lr=0.01, max_iter=100):
        """Fit on single client's validation data."""
        optimizer = torch.optim.LBFGS([self.A, self.B], lr=lr, max_iter=max_iter)

        def closure():
            optimizer.zero_grad()
            out = self.forward(logits)
            loss = F.binary_cross_entropy(out, labels)
            loss.backward()
            return loss

        optimizer.step(closure)


class BetaCalibrator(nn.Module):
    """
    Beta Calibration for binary classification.
    
    More flexible than Platt Scaling with 3 parameters (a, b, c):
    P_calibrated = sigmoid(a + b * log(p / (1-p)))^c
    
    Reference: Kull et al. (2017) "Beta calibration: a well-founded and
    easily implemented improvement on logistic calibration for binary classifiers"
    
    Benefits over Platt:
    - Handles asymmetric miscalibration
    - More flexible shape (shape parameter c)
    - Still resistant to overfitting (only 3 params)
    """

    def __init__(self):
        super(BetaCalibrator, self).__init__()
        self.a = nn.Parameter(torch.zeros(1))  # Intercept
        self.b = nn.Parameter(torch.ones(1))   # Slope
        self.c = nn.Parameter(torch.ones(1))   # Shape (c=1 reduces to Platt)

    def forward(self, logits):
        # Convert logits to probabilities first
        probs = torch.sigmoid(logits)
        
        # Clamp to avoid log(0)
        probs = torch.clamp(probs, 1e-7, 1 - 1e-7)
        
        # Log-odds transformation with shape parameter
        log_odds = torch.log(probs / (1 - probs))
        
        # Beta calibration: sigmoid(a + b * log_odds) with shape c
        # When c=1, this is equivalent to Platt scaling
        calibrated_logits = self.a + self.b * log_odds
        
        # Apply shape parameter (c controls asymmetry)
        # For c != 1, we apply power transformation in probability space
        calibrated_probs = torch.sigmoid(calibrated_logits)
        
        # Shape adjustment (simplified beta-like transformation)
        if self.c.item() != 1.0:
            calibrated_probs = torch.pow(calibrated_probs, self.c)
            
        return calibrated_probs

    def fit(self, logits, labels, lr=0.01, max_iter=100):
        """Fit beta calibrator on validation data."""
        optimizer = torch.optim.LBFGS([self.a, self.b, self.c], lr=lr, max_iter=max_iter)

        def closure():
            optimizer.zero_grad()
            out = self.forward(logits)
            # Clamp output to avoid BCE issues
            out = torch.clamp(out, 1e-7, 1 - 1e-7)
            loss = F.binary_cross_entropy(out, labels)
            loss.backward()
            return loss

        optimizer.step(closure)
        print(f"  Beta Parameters: a={self.a.item():.4f}, b={self.b.item():.4f}, c={self.c.item():.4f}")
