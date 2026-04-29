"""
Griffin model adapted for Bitcoin anomaly detection.

Same RG-LRU architecture as the ECG PoC, but tuned for financial time-series:
- 8 input features (returns, volume, volatility, momentum)
- Next-step prediction: learns "normal" BTC tick patterns
- Anomaly score = prediction error (z-score vs training baseline)
- Early warning head: P(large move in next N ticks)

The key insight: Griffin processes each tick with O(1) memory,
maintaining a compressed "market state" in its hidden state.
When the market starts behaving unusually, prediction error spikes
BEFORE the anomaly fully develops.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

# Reuse the core Griffin blocks from the ECG PoC
from griffin_model import RGLRU, GriffinBlock


class GriffinBTCDetector(nn.Module):
    """
    Griffin model for Bitcoin anomaly detection.

    Architecture:
        Input (8 features) → Linear → [GriffinBlock × n_layers] → Linear → Output (8 features)

    Two output heads:
        1. Next-step prediction (reconstruction) — always active
        2. Early warning head: P(anomaly in next N ticks) — trained on labeled events
    """

    INPUT_DIM = 8  # 8 financial features

    def __init__(self, d_model: int = 64, n_layers: int = 3, early_warning_window: int = 60):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        self.early_warning_window = early_warning_window

        # Input projection
        self.input_proj = nn.Linear(self.INPUT_DIM, d_model)

        # Griffin blocks
        self.layers = nn.ModuleList([
            GriffinBlock(d_model) for _ in range(n_layers)
        ])

        # Next-step prediction head
        self.pred_head = nn.Linear(d_model, self.INPUT_DIM)

        # Early warning head: P(significant move in next N ticks)
        self.warning_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )

        # Training baseline for anomaly scoring
        self.register_buffer("baseline_mean", torch.tensor(0.0))
        self.register_buffer("baseline_std", torch.tensor(1.0))

    def forward(self, x, hidden_states=None):
        """
        Args:
            x: (batch, seq_len, INPUT_DIM)
            hidden_states: list of hidden states per layer

        Returns:
            pred: (batch, seq_len, INPUT_DIM) — next-step predictions
            warning: (batch, seq_len, 1) — P(anomaly in next N ticks)
            new_hidden: list of hidden states
        """
        h = self.input_proj(x)

        if hidden_states is None:
            hidden_states = [None] * self.n_layers

        new_hidden = []
        for layer, hs in zip(self.layers, hidden_states):
            h, hs_new = layer(h, hs)
            new_hidden.append(hs_new)

        pred = self.pred_head(h)
        warning = torch.sigmoid(self.warning_head(h))

        return pred, warning, new_hidden

    @torch.no_grad()
    def detect_realtime(self, x, hidden_states=None):
        """
        Real-time anomaly detection — process tick by tick.
        This is the O(1) memory inference path.

        Args:
            x: (batch, seq_len, INPUT_DIM) — full sequence to process

        Returns:
            anomaly_scores: list of floats (one per tick)
            warning_probs: list of floats (P(event) per tick)
            hidden_states: final hidden states
        """
        self.eval()
        anomaly_scores = []
        warning_probs = []

        for t in range(x.size(1) - 1):
            x_t = x[:, t:t+1, :]  # (B, 1, D)
            pred, warning, hidden_states = self(x_t, hidden_states)

            # Anomaly score = prediction error vs next tick
            target = x[:, t+1:t+2, :]
            mse = F.mse_loss(pred, target, reduction="none").mean(dim=-1).mean(dim=-1)

            # Z-score normalization
            z_score = (mse - self.baseline_mean) / (self.baseline_std + 1e-8)
            anomaly_scores.append(z_score.item())
            warning_probs.append(warning.squeeze().item())

        return anomaly_scores, warning_probs, hidden_states

    def set_baseline(self, losses: list[float]):
        """Store training loss statistics for z-score normalization."""
        arr = np.array(losses)
        self.baseline_mean = torch.tensor(float(arr.mean()))
        self.baseline_std = torch.tensor(float(arr.std() + 1e-8))


class TransformerBTCDetector(nn.Module):
    """
    Transformer baseline for comparison.
    Standard multi-head self-attention — O(n²) memory.
    """

    INPUT_DIM = 8

    def __init__(self, d_model: int = 64, n_layers: int = 3, n_heads: int = 4):
        super().__init__()
        self.d_model = d_model
        self.input_proj = nn.Linear(self.INPUT_DIM, d_model)

        # Positional encoding
        self.pos_encoding = nn.Parameter(torch.randn(1, 4096, d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.pred_head = nn.Linear(d_model, self.INPUT_DIM)
        self.warning_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )

        self.register_buffer("baseline_mean", torch.tensor(0.0))
        self.register_buffer("baseline_std", torch.tensor(1.0))

    def forward(self, x, hidden_states=None):
        B, T, D = x.shape
        h = self.input_proj(x) + self.pos_encoding[:, :T, :]

        # Causal mask
        mask = nn.Transformer.generate_square_subsequent_mask(T, device=x.device)
        h = self.encoder(h, mask=mask, is_causal=True)

        pred = self.pred_head(h)
        warning = torch.sigmoid(self.warning_head(h))
        return pred, warning, None

    def set_baseline(self, losses: list[float]):
        arr = np.array(losses)
        self.baseline_mean = torch.tensor(float(arr.mean()))
        self.baseline_std = torch.tensor(float(arr.std() + 1e-8))
