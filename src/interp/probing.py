"""
-------------------------------------------------
Probing classifiers for interpretability research.

Contains:
- ProbingClassifier: Binary linear probe (logistic-regression) using sklearn
- ProbabilisticProbe: PyTorch probe for predicting soft probability targets (BCE loss)

Key features
------------
• Pure NumPy / PyTorch data interface – no custom dataset or activation classes
• Optional z-score normalisation (on the training set)
• ProbingClassifier uses scikit-learn's LogisticRegression (L2-regularised)
• ProbabilisticProbe uses PyTorch for gradient-based training with BCE loss

"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, Union

import einops
import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
    precision_recall_curve,
    auc,
)
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


ArrayLike = Union[np.ndarray, torch.Tensor]


# --------------------------------------------------------------------------- #
# Core classifier
# --------------------------------------------------------------------------- #
class ProbingClassifier:
    """Binary linear probe trained with logistic regression."""

    def __init__(
        self,
        reg_coeff: float = 1e3,
        normalize: bool = True,
        fit_intercept: bool = False,
        dtype: torch.dtype = torch.float32,
        pca_components: int | None = None,
        class_weight: str | None = None,
    ) -> None:
        self.reg_coeff = float(reg_coeff)
        self.normalize = bool(normalize)
        self.fit_intercept = bool(fit_intercept)
        self.dtype = dtype
        self.apply_pca = bool(pca_components is not None)
        self.pca_components = pca_components
        self.class_weight = class_weight

        self.direction: torch.Tensor | None = None  # shape (hidden_dim,)
        self.intercept: torch.Tensor | None = None  # shape (1,)
        self.scaler_mean: torch.Tensor | None = None
        self.scaler_std: torch.Tensor | None = None
        self.pca: PCA | None = None

    # ----------------------------- training -------------------------------- #
    def fit(self, pos_acts: ArrayLike, neg_acts: ArrayLike) -> "ProbingClassifier":
        """Fit the probe on positive/negative token activations.

        Parameters:
        pos_acts: (n_pos, hidden_dim)
        neg_acts: (n_neg, hidden_dim)

        Returns:
        self
        """
        X_pos = _to_numpy(pos_acts, self.dtype)
        X_neg = _to_numpy(neg_acts, self.dtype)

        if X_pos.ndim != 2 or X_neg.ndim != 2:
            raise ValueError(
                f"Activations must be 2-D: (n_tokens, hidden_dim). "
                f"Got {X_pos.shape} and {X_neg.shape}."
            )

        X = np.vstack([X_pos, X_neg])
        y = np.concatenate(
            [
                np.ones(X_pos.shape[0], dtype=np.int64),
                np.zeros(X_neg.shape[0], dtype=np.int64),
            ]
        )

        # Normalisation ---------------------------------------------------- #
        if self.normalize:
            scaler = StandardScaler()
            X = scaler.fit_transform(X)  # type: ignore[arg-type]

            self.scaler_mean = torch.tensor(scaler.mean_, dtype=self.dtype)
            self.scaler_std = torch.tensor(scaler.scale_, dtype=self.dtype)
        else:
            # Identity transform
            self.scaler_mean = torch.zeros(X.shape[1], dtype=self.dtype)
            self.scaler_std = torch.ones(X.shape[1], dtype=self.dtype)

        if self.apply_pca:
            self.pca = PCA(n_components=self.pca_components)
            X = self.pca.fit_transform(X)

        # Logistic regression --------------------------------------------- #
        model = LogisticRegression(
            C=1.0 / self.reg_coeff,
            fit_intercept=self.fit_intercept,
            random_state=42,
            solver="liblinear",
            class_weight=self.class_weight,
        )
        model.fit(X, y)  # type: ignore[arg-type]

        self.direction = torch.tensor(model.coef_.reshape(-1), dtype=self.dtype)
        self.intercept = torch.tensor(model.intercept_, dtype=self.dtype)
        return self

    # ------------------------------ scoring -------------------------------- #
    @torch.no_grad()
    def score_tokens(
        self, hidden_states: ArrayLike, apply_sigmoid: bool = False
    ) -> torch.Tensor:
        """
        Get per-token *logit* scores for a single sequence.

        Parameters:
        hidden_states : (seq_len, hidden_dim)

        Returns:
        torch.Tensor
            1-D tensor of shape (seq_len,) where higher → more 'positive'.
        """
        if self.direction is None:
            raise RuntimeError("Probe not trained – call .fit(...) first.")

        hs = _to_torch(hidden_states, self.dtype)

        if hs.ndim != 2:
            raise ValueError(
                f"hidden_states must be (seq_len, hidden_dim); got {hs.shape}."
            )

        if self.normalize:
            hs = (hs - self.scaler_mean) / self.scaler_std  # type: ignore[arg-type]

        if self.apply_pca:
            hs = _to_numpy(hs, self.dtype)
            hs = self.pca.transform(hs)
            hs = _to_torch(hs, self.dtype)

        logits = torch.einsum("ih,h->i", hs, self.direction)  # (seq_len,)
        if self.fit_intercept:
            logits += self.intercept

        if apply_sigmoid:
            return torch.sigmoid(logits)
        else:
            return logits

    def evaluate(self, pos_acts: ArrayLike, neg_acts: ArrayLike) -> Dict[str, float]:
        """
        Evaluate the probe on test data.

        Parameters:
        pos_acts: (n_pos, hidden_dim) - positive test activations
        neg_acts: (n_neg, hidden_dim) - negative test activations

        Returns:
        Dict[str, float]
            Dictionary containing evaluation metrics (accuracy, precision, recall, f1, roc_auc, brier_score, log_loss)
        """
        if self.direction is None:
            raise RuntimeError("Probe not trained – call .fit(...) first.")

        if len(pos_acts) == 0 or len(neg_acts) == 0:
            print("No test data to evaluate on.")
            return {}

        # Get scores for test data
        pos_scores = self.score_tokens(pos_acts).cpu().numpy()
        neg_scores = self.score_tokens(neg_acts).cpu().numpy()

        # Get probabilities for probabilistic metrics
        pos_probs = self.score_tokens(pos_acts, apply_sigmoid=True).cpu().numpy()
        neg_probs = self.score_tokens(neg_acts, apply_sigmoid=True).cpu().numpy()

        # Create labels and predictions
        y_true = np.concatenate([np.ones(len(pos_scores)), np.zeros(len(neg_scores))])
        y_scores = np.concatenate([pos_scores, neg_scores])
        y_probs = np.concatenate([pos_probs, neg_probs])
        y_pred = (y_scores > 0).astype(int)  # threshold at 0

        prec, rec, _ = precision_recall_curve(y_true, y_scores)
        pr_auc = auc(rec, prec)

        prec0, rec0, _ = precision_recall_curve(y_true, y_scores, pos_label=0)
        pr_auc0 = auc(rec0, prec0)

        # Calculate metrics
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred),
            "recall": recall_score(y_true, y_pred),
            "f1_score": f1_score(y_true, y_pred),
            "f1_score_of_0": f1_score(y_true, y_pred, pos_label=0),
            "roc_auc": roc_auc_score(y_true, y_scores),
            "brier_score": brier_score_loss(y_true, y_probs),
            "log_loss": log_loss(y_true, y_probs),
            "pr_auc": pr_auc,
            "pr_auc_of_0": pr_auc0,
        }

        return metrics

    # ----------------------------- I/O helpers ---------------------------- #
    def save(self, file: str | Path) -> None:
        """Pickle the probe to *file*."""
        payload = {
            "direction": self.direction.cpu() if self.direction is not None else None,
            "intercept": self.intercept.cpu() if self.intercept is not None else None,
            "scaler_mean": self.scaler_mean.cpu()
            if self.scaler_mean is not None
            else None,
            "scaler_std": self.scaler_std.cpu()
            if self.scaler_std is not None
            else None,
            "pca": self.pca,
            "reg_coeff": self.reg_coeff,
            "normalize": self.normalize,
            "fit_intercept": self.fit_intercept,
            "dtype": self.dtype,
            "apply_pca": self.apply_pca,
            "pca_components": self.pca_components,
            "class_weight": self.class_weight,
        }
        with open(file, "wb") as f:
            pickle.dump(payload, f)

    @classmethod
    def load(cls, file: str | Path) -> "ProbingClassifier":
        """Load a probe previously saved with :pymeth:`save`."""
        with open(file, "rb") as f:
            payload = pickle.load(f)

        obj = cls(
            reg_coeff=payload["reg_coeff"],
            normalize=payload["normalize"],
            fit_intercept=payload.get("fit_intercept", False),
            dtype=payload["dtype"],
            pca_components=payload.get("pca_components", None),
            class_weight=payload.get("class_weight", None),
        )
        obj.direction = payload["direction"]
        obj.intercept = payload.get(
            "intercept", torch.tensor(0.0, dtype=payload["dtype"])
        )
        obj.scaler_mean = payload["scaler_mean"]
        obj.scaler_std = payload["scaler_std"]
        obj.pca = payload.get("pca", None)
        return obj


# --------------------------------------------------------------------------- #
# ProbabilisticProbe - PyTorch probe for soft probability targets
# --------------------------------------------------------------------------- #


class ProbabilisticProbe(nn.Module):
    """
    Probe for predicting probabilities from activations.

    Unlike ProbingClassifier which uses sklearn for binary classification,
    this probe uses PyTorch and BCE loss for predicting soft probability
    targets (values in [0, 1]).

    Can be either a linear probe or an MLP depending on hidden_dims.

    Architecture: [Linear] or [Linear -> ReLU -> ... -> Linear] (outputs logits)
    Loss: BCEWithLogitsLoss (numerically stable)
    """

    def __init__(self, input_size: int, hidden_dims: list[int] | None = None):
        """
        Args:
            input_size: dimension of input activations
            hidden_dims: list of hidden layer sizes for MLP, or None for linear probe
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_dims = hidden_dims

        # Build network
        if hidden_dims is None or len(hidden_dims) == 0:
            # Linear probe
            self.network = nn.Linear(input_size, 1)
        else:
            # MLP probe
            layers = []
            prev_dim = input_size
            for dim in hidden_dims:
                layers.append(nn.Linear(prev_dim, dim))
                layers.append(nn.ReLU())
                prev_dim = dim
            layers.append(nn.Linear(prev_dim, 1))
            self.network = nn.Sequential(*layers)

        # Normalization buffers: identity transform (mean=0, std=1) until set
        # This ensures buffers are always in state_dict for save/load compatibility
        self.register_buffer("mean", torch.zeros(input_size))
        self.register_buffer("std", torch.ones(input_size))

    def set_normalization(self, mean: torch.Tensor, std: torch.Tensor, device: torch.device) -> None:
        """Set normalization parameters (computed from training data)."""
        # Re-register buffers with actual values (cleaner than direct assignment)
        self.register_buffer("mean", mean.to(device))
        self.register_buffer("std", std.to(device))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: normalize -> network. Returns logits."""
        x = (x - self.mean) / self.std
        return self.network(x).squeeze(-1)

    def predict_proba(self, x: torch.Tensor | np.ndarray) -> np.ndarray:
        """Predict probabilities (applies sigmoid to logits)."""
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        self.eval()
        with torch.no_grad():
            logits = self(x)
            return torch.sigmoid(logits).cpu().numpy()

    def predict_proba_tensor(self, x: torch.Tensor) -> torch.Tensor:
        """Predict probabilities and keep them as a torch tensor."""
        self.eval()
        with torch.no_grad():
            logits = self(x)
            return torch.sigmoid(logits)

    def save(self, path: Path | str) -> None:
        """Save probe to file."""
        torch.save(
            {
                "probe_type": "probabilistic",
                "state_dict": self.state_dict(),
                "input_size": self.input_size,
                "hidden_dims": self.hidden_dims,
            },
            path,
        )

    @classmethod
    def load(cls, path: Path | str) -> "ProbabilisticProbe":
        """Load probe from file."""
        data = torch.load(path, map_location="cpu", weights_only=False)
        probe = cls(data["input_size"], data["hidden_dims"])
        probe.load_state_dict(data["state_dict"])
        return probe


# --------------------------------------------------------------------------- #
# Attention Probe - PyTorch probe that takes variable length input sequences,
# aggregates the activation with attention and predicts a probability.
# --------------------------------------------------------------------------- #

class AttentionProbe(nn.Module):
    def __init__(self, input_size: int, num_heads: int, hidden_dims: list[int] | None = None, attn_hidden_dims: list[int] | None = None, dropout: float = 0.0, use_position_bias: bool = False):
        """
        Args:
            input_size: dimension of input activations (hidden_size)
            hidden_dims: list of hidden layer sizes for MLP, or None for linear probe
            attn_hidden_dims: list of hidden layer sizes for query MLP, or None for linear
            dropout: dropout probability applied after each ReLU in both MLPs
            use_position_bias: if True, add a learnable linear position bias to attention logits
        """
        super().__init__()
        self.input_size = input_size
        self.num_heads = num_heads
        self.hidden_dims = hidden_dims
        self.attn_hidden_dims = attn_hidden_dims
        self.dropout = dropout
        self.use_position_bias = use_position_bias

        # Build value projection
        if hidden_dims is None or len(hidden_dims) == 0:
            self.value_projection = nn.Linear(input_size, num_heads)
        else:
            value_layers = []
            prev_dim = input_size
            for dim in hidden_dims:
                value_layers.append(nn.Linear(prev_dim, dim))
                value_layers.append(nn.ReLU())
                if dropout > 0:
                    value_layers.append(nn.Dropout(dropout))
                prev_dim = dim

            value_layers.append(nn.Linear(prev_dim, num_heads))
            self.value_projection = nn.Sequential(*value_layers)
        
        # Build query projection
        if attn_hidden_dims is None or len(attn_hidden_dims) == 0:
            self.query_projection = nn.Linear(input_size, num_heads)
        else:
            query_layers = []
            prev_dim = input_size
            for dim in attn_hidden_dims:
                query_layers.append(nn.Linear(prev_dim, dim))
                query_layers.append(nn.ReLU())
                if dropout > 0:
                    query_layers.append(nn.Dropout(dropout))
                prev_dim = dim

            query_layers.append(nn.Linear(prev_dim, num_heads))
            self.query_projection = nn.Sequential(*query_layers)
        
        # Learnable linear position bias: weight * (pos / seq_len) + offset per head
        if use_position_bias:
            self.position_bias_weight = nn.Parameter(torch.ones(num_heads))
            self.position_bias_offset = nn.Parameter(torch.zeros(num_heads))

        # Per-feature normalization buffers (identity transform until set)
        self.register_buffer("mean", torch.zeros(input_size))
        self.register_buffer("std", torch.ones(input_size))

    def set_normalization(self, mean: torch.Tensor, std: torch.Tensor, device: torch.device) -> None:
        """Set per-feature normalization parameters (computed from training data)."""
        self.register_buffer("mean", mean.to(device))
        self.register_buffer("std", std.to(device))

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, hidden_size)
            mask: (batch_size, seq_len) - mask for the input sequence (because the sequence is variable length)

        Returns:
            (batch_size,) - probability
        """
        x = (x - self.mean) / self.std

        values = self.value_projection(x)  # (batch_size, seq_len, num_heads)
        attention_logits = self.query_projection(x)  # (batch_size, seq_len, num_heads) here the queries are also attention_logits

        if self.use_position_bias:
            batch_size, seq_len, _ = x.shape
            positions = torch.arange(seq_len, device=x.device, dtype=x.dtype)  # (seq_len,)
            seq_lengths = mask.sum(dim=-1, keepdim=True).clamp(min=1)  # (batch_size, 1)
            norm_positions = positions.unsqueeze(0) # (batch_size, seq_len) in [0, 1)
            pos_bias = norm_positions.unsqueeze(-1) * self.position_bias_weight + self.position_bias_offset  # (batch_size, seq_len, num_heads)
            attention_logits = attention_logits + pos_bias

        mask = einops.repeat(mask, "batch seq_len -> batch seq_len num_heads", num_heads=self.num_heads)
        masked_attention_logits = attention_logits.masked_fill(~mask, float("-inf"))

        attention_weights = nn.functional.softmax(masked_attention_logits, dim=-2)  # (batch_size, seq_len, num_heads)

        per_head_output = einops.einsum(attention_weights, values, "batch seq_len num_heads, batch seq_len num_heads-> batch num_heads")

        # Sum the head outputs
        output = einops.reduce(per_head_output, "batch num_heads -> batch", "sum")
        
        return output
    
    def predict_proba(self, x: torch.Tensor | np.ndarray, mask: torch.Tensor) -> np.ndarray:
        """Predict probabilities (applies sigmoid to logits)."""
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        self.eval()
        with torch.no_grad():
            logits = self(x, mask)
            return torch.sigmoid(logits).numpy()

    def save(self, path: Path | str) -> None:
        """Save probe to file."""
        torch.save(
            {
                "probe_type": "attention",
                "state_dict": self.state_dict(),
                "input_size": self.input_size,
                "num_heads": self.num_heads,
                "hidden_dims": self.hidden_dims,
                "attn_hidden_dims": self.attn_hidden_dims,
                "dropout": self.dropout,
                "use_position_bias": self.use_position_bias,
            },
            path,
        )

    @classmethod
    def load(cls, path: Path | str) -> "AttentionProbe":
        """Load probe from file."""
        data = torch.load(path, map_location="cpu", weights_only=False)
        probe = cls(
            data["input_size"],
            data.get("num_heads", 1),
            data["hidden_dims"],
            data["attn_hidden_dims"],
            data.get("dropout", 0.0),
            data.get("use_position_bias", False),
        )
        probe.load_state_dict(data["state_dict"])
        return probe


# --------------------------------------------------------------------------- #
# GDMAttention Probe - PyTorch probe that takes variable length input sequences,
# aggregates the activation with attention and predicts a probability.
# This one is based on the architecture described in https://arxiv.org/pdf/2601.11516 
# --------------------------------------------------------------------------- #

class GDMAttentionProbe(nn.Module):
    def __init__(self, input_size: int, hidden_dims: list[int] | None = None):
        """
        This probe has a common MLP layer for each input token, and its output then goes through a query and value projection.
        Args:
            input_size: dimension of input activations (hidden_size)
            hidden_dims: list of hidden layer sizes for MLP, or None for linear probe
            attn_hidden_dims: list of hidden layer sizes for query MLP, or None for linear
            dropout: dropout probability applied after each ReLU in both MLPs
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_dims = hidden_dims

        # Build mlp layer
        if hidden_dims is None or len(hidden_dims) == 0:
            raise ValueError("hidden_dims must be provided")
        else:
            mlp_layers = []
            prev_dim = input_size
            for dim in hidden_dims:
                mlp_layers.append(nn.Linear(prev_dim, dim))
                mlp_layers.append(nn.ReLU())
                prev_dim = dim

            self.mlp_layer = nn.Sequential(*mlp_layers)
        
        self.query_projection = nn.Linear(hidden_dims[-1], 1)
        self.value_projection = nn.Linear(hidden_dims[-1], 1)



        self.position_weight = nn.Parameter(torch.ones(1))

        # Per-feature normalization buffers (identity transform until set)
        self.register_buffer("mean", torch.zeros(input_size))
        self.register_buffer("std", torch.ones(input_size))

    def set_normalization(self, mean: torch.Tensor, std: torch.Tensor, device: torch.device) -> None:
        """Set per-feature normalization parameters (computed from training data)."""
        self.register_buffer("mean", mean.to(device))
        self.register_buffer("std", std.to(device))

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, hidden_size)
            mask: (batch_size, seq_len) - mask for the input sequence (because the sequence is variable length)

        Returns:
            (batch_size,) - probability
        """
        x = (x - self.mean) / self.std

        y = self.mlp_layer(x)  # (batch_size, seq_len, last_hidden_dim)

        attention_logits = self.query_projection(y)  # (batch_size, seq_len, 1) here the queries are also attention_logits

        values = self.value_projection(y)  # (batch_size, seq_len, 1)

        mask = einops.rearrange(mask, "batch seq_len -> batch seq_len 1")
        masked_attention_logits = attention_logits.masked_fill(~mask, float("-inf"))

        attention_weights = nn.functional.softmax(masked_attention_logits, dim=-2)  # (batch_size, seq_len, 1)

        attention_weights = einops.rearrange(attention_weights, "batch seq_len 1 -> batch seq_len")
        values = einops.rearrange(values, "batch seq_len 1 -> batch seq_len")

        output = einops.einsum(attention_weights, values, "batch seq_len, batch seq_len -> batch ")

        return output
    
    def predict_proba(self, x: torch.Tensor | np.ndarray, mask: torch.Tensor) -> np.ndarray:
        """Predict probabilities (applies sigmoid to logits)."""
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        self.eval()
        with torch.no_grad():
            logits = self(x, mask)
            return torch.sigmoid(logits).numpy()

    def save(self, path: Path | str) -> None:
        """Save probe to file."""
        torch.save(
            {
                "probe_type": "gdm_attention",
                "state_dict": self.state_dict(),
                "input_size": self.input_size,
                "hidden_dims": self.hidden_dims,
            },
            path,
        )

    @classmethod
    def load(cls, path: Path | str) -> "GDMAttentionProbe":
        """Load probe from file."""
        data = torch.load(path, map_location="cpu", weights_only=False)
        probe = cls(data["input_size"], data["hidden_dims"])
        probe.load_state_dict(data["state_dict"])
        return probe



# --------------------------------------------------------------------------- #
# Formatting Helpers
# --------------------------------------------------------------------------- #


def format_response_with_probabilities(
    sentences: list[str], probabilities: list[float]
) -> str:
    """Format response with probability annotations.

    Example: "(0.6) Sentence 1. (0.8) Sentence 2. (1.0)"

    The first probability is the base (before any response).
    Each subsequent probability is after that sentence.

    Args:
        sentences: List of sentences in the response
        probabilities: List of probabilities (len = len(sentences) + 1)
                      First is base probability, rest are after each sentence

    Returns:
        Formatted string with probabilities interleaved
    """
    if not sentences:
        return f"({probabilities[0]:.2f})"

    parts = [f"({probabilities[0]:.2f})"]  # Base probability
    for i, sentence in enumerate(sentences):
        parts.append(f" {sentence}")
        if i + 1 < len(probabilities):
            parts.append(f" ({probabilities[i + 1]:.2f})")

    return "".join(parts)


# --------------------------------------------------------------------------- #
# Utils
# --------------------------------------------------------------------------- #
def _load_acts(path) -> torch.Tensor:
    """
    Load activations from a .npy, .pt, or .pth file. Accepts str or Path.
    Returns the activations tensor, handling both old and new formats.
    """
    if not isinstance(path, Path):
        path = Path(path)
    if path.suffix == ".npy":
        return torch.from_numpy(np.load(path))
    if path.suffix in {".pt", ".pth"}:
        # First try loading with weights_only=False to handle dict format
        try:
            data = torch.load(path, map_location="cpu", weights_only=False)
            if isinstance(data, dict) and "activations" in data:
                # New format with metadata
                return data["activations"]
            elif isinstance(data, torch.Tensor):
                # Old direct tensor format
                return data
            else:
                # Try with weights_only=True for safety
                return torch.load(path, map_location="cpu", weights_only=True)
        except Exception:
            # Fallback to weights_only=True
            return torch.load(path, map_location="cpu", weights_only=True)
    raise ValueError(f"Unsupported file format: {path}")


def _to_numpy(x: ArrayLike, dtype: torch.dtype = torch.float32) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().to(dtype).cpu().numpy()
    if isinstance(x, np.ndarray):
        return x.astype(np.float32)
    raise TypeError(f"Unsupported type {type(x)}; expected torch.Tensor or np.ndarray.")


def _to_torch(x: ArrayLike, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.to(dtype).cpu()
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).to(dtype)
    raise TypeError(f"Unsupported type {type(x)}; expected torch.Tensor or np.ndarray.")
