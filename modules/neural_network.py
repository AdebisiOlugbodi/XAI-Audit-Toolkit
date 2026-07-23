"""
modules/neural_network.py

A simple but robust PyTorch neural network for recruitment CV screening.
Wrapped in a fully sklearn-compatible interface so it plugs directly into
the existing pipeline without any changes to other modules.

Architecture:
  Input → Dense(64) → BatchNorm → ReLU → Dropout(0.3)
        → Dense(32) → BatchNorm → ReLU → Dropout(0.2)
        → Dense(16) → ReLU
        → Dense(1)  → Sigmoid

Install PyTorch:
  pip install torch
"""

import numpy as np
import os
import pandas as pd

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted


# Network Definition 

if TORCH_AVAILABLE:
    class _RecruitmentNet(nn.Module):
        """
        Feedforward neural network for binary recruitment classification.
        3-hidden-layer architecture with BatchNorm and Dropout.
        """

        def __init__(self, input_dim: int):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, 64),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.Dropout(0.3),

                nn.Linear(64, 32),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.Dropout(0.2),

                nn.Linear(32, 16),
                nn.ReLU(),

                nn.Linear(16, 1),
            )

        def forward(self, x):
            return self.net(x).squeeze(1)


# Sklearn-Compatible Wrapper 

class PyTorchNeuralNetwork(ClassifierMixin, BaseEstimator):
    """
    Fully sklearn-compatible PyTorch neural network classifier.

    Inherits ClassifierMixin BEFORE BaseEstimator — this is the correct
    order required by sklearn 1.6+ for is_classifier() to return True.

    Parameters
    ----------
    epochs : int     — training passes (default 100)
    lr : float       — Adam learning rate (default 0.001)
    batch_size : int — mini-batch size (default 32)
    threshold : float — decision threshold (default 0.5)
    verbose : bool   — print loss every 10 epochs (default True)
    """

    # tells sklearn explicitly this is a classifier
    _estimator_type = "classifier"

    def __init__(
        self,
        epochs: int      = 100,
        lr: float        = 0.001,
        batch_size: int  = 32,
        threshold: float = 0.5,
        verbose: bool    = True,
    ):
        # BaseEstimator requires __init__ params to match attribute names exactly
        # Do NOT set any non-param attributes here
        self.epochs     = epochs
        self.lr         = lr
        self.batch_size = batch_size
        self.threshold  = threshold
        self.verbose    = verbose

    # Training 

    def fit(self, X, y):
        """Train the network. Sets fitted attributes with trailing underscore."""
        if not TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch is not installed.\n"
                "Install with: pip install torch"
            )

        X_arr = self._to_array(X)
        y_arr = np.array(y, dtype=np.float32)

        # Required by sklearn: store classes seen during fit
        self.classes_    = np.unique(y_arr).astype(int)
        self.n_features_in_ = X_arr.shape[1]
        self.input_dim_  = X_arr.shape[1]

        self.device_ = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        X_t = torch.tensor(X_arr, dtype=torch.float32)
        y_t = torch.tensor(y_arr, dtype=torch.float32)

        self.net_ = _RecruitmentNet(self.input_dim_).to(self.device_)

        loader = DataLoader(
            TensorDataset(X_t, y_t),
            batch_size=self.batch_size,
            shuffle=True,
        )

        # Weighted BCE loss for class imbalance
        n_neg = float((y_t == 0).sum())
        n_pos = float((y_t == 1).sum().clamp(min=1))
        pos_weight = torch.tensor([n_neg / n_pos]).to(self.device_)
        criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        optimiser = torch.optim.Adam(
            self.net_.parameters(), lr=self.lr, weight_decay=1e-4
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimiser, patience=10, factor=0.5
        )

        print(
            f"[Neural Network] Training on {self.device_} | "
            f"Epochs: {self.epochs} | Batch: {self.batch_size} | LR: {self.lr}"
        )

        self.training_history_ = []
        self.net_.train()
        avg_loss = 0.0

        for epoch in range(1, self.epochs + 1):
            epoch_loss = 0.0
            for X_batch, y_batch in loader:
                X_batch = X_batch.to(self.device_)
                y_batch = y_batch.to(self.device_)
                optimiser.zero_grad()
                loss = criterion(self.net_(X_batch), y_batch)
                loss.backward()
                optimiser.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(loader)
            self.training_history_.append(avg_loss)
            scheduler.step(avg_loss)

            if self.verbose and epoch % 10 == 0:
                print(f"  Epoch {epoch:>3}/{self.epochs}  loss: {avg_loss:.4f}")

        self.net_.eval()

        # Compute internal permutation importances (used as fallback by ExplainabilityEngine)
        self.feature_importances_ = self._compute_importances(X_t, y_t)

        print(f"[Neural Network] Training complete. Final loss: {avg_loss:.4f}")
        return self

    # Inference 

    def predict_proba(self, X):
        """
        Returns array of shape (n_samples, 2).
        Column 0 = P(rejected), Column 1 = P(hired).
        """
        check_is_fitted(self, ["net_", "classes_"])
        probs = self._sigmoid(X)
        return np.column_stack([1.0 - probs, probs])

    def predict(self, X):
        """Returns binary predictions (0 or 1) using self.threshold."""
        check_is_fitted(self, ["net_", "classes_"])
        probs = self._sigmoid(X)
        return (probs >= self.threshold).astype(int)

    def _sigmoid(self, X) -> np.ndarray:
        """Run forward pass and apply sigmoid to get probabilities."""
        self.net_.eval()
        X_t = torch.tensor(self._to_array(X), dtype=torch.float32).to(self.device_)
        with torch.no_grad():
            logits = self.net_(X_t).cpu().numpy()
        # Apply sigmoid manually (net outputs raw logits)
        return 1.0 / (1.0 + np.exp(-logits))

    #  Feature Importances 

    def _compute_importances(self, X_t, y_t) -> np.ndarray:
        """
        Internal permutation importance computed on training data.
        Stored as feature_importances_ for compatibility with ExplainabilityEngine.
        """
        self.net_.eval()
        importances = np.zeros(X_t.shape[1])

        with torch.no_grad():
            base_preds = (self._sigmoid(X_t.numpy()) >= self.threshold)
            base_acc   = (base_preds == y_t.numpy()).mean()

            for i in range(X_t.shape[1]):
                X_shuf = X_t.clone()
                idx    = torch.randperm(X_shuf.shape[0])
                X_shuf[:, i] = X_shuf[idx, i]
                preds  = (self._sigmoid(X_shuf.numpy()) >= self.threshold)
                drop   = base_acc - (preds == y_t.numpy()).mean()
                importances[i] = max(0.0, drop)

        total = importances.sum()
        return importances / total if total > 0 else importances

    # Persistence 

    def save(self, path: str):
        """Save model weights to a .pt file."""
        check_is_fitted(self, ["net_"])
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        torch.save({
            "state_dict":   self.net_.state_dict(),
            "input_dim":    self.input_dim_,
            "classes":      self.classes_,
            "importances":  self.feature_importances_,
            "history":      self.training_history_,
            "params":       self.get_params(),
        }, path)
        print(f"[Neural Network] Model saved: {path}")

    def load(self, path: str):
        """Load model weights from a .pt file."""
        self.device_ = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(path, map_location=self.device_)
        self.input_dim_          = ckpt["input_dim"]
        self.n_features_in_      = ckpt["input_dim"]
        self.classes_            = ckpt["classes"]
        self.feature_importances_ = ckpt["importances"]
        self.training_history_   = ckpt.get("history", [])
        self.net_ = _RecruitmentNet(self.input_dim_).to(self.device_)
        self.net_.load_state_dict(ckpt["state_dict"])
        self.net_.eval()
        print(f"[Neural Network] Model loaded: {path}")
        return self

    # Helpers 

    def _to_array(self, X) -> np.ndarray:
        if isinstance(X, pd.DataFrame):
            return X.values.astype(np.float32)
        return np.array(X, dtype=np.float32)

    def __repr__(self):
        return (
            f"PyTorchNeuralNetwork(epochs={self.epochs}, lr={self.lr}, "
            f"batch_size={self.batch_size}, threshold={self.threshold})"
        )
