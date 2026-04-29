"""
Griffin RG-LRU Model — simplified implementation for PoC.
Based on the RecurrentGemma architecture (Google DeepMind, 2024).

Key idea: Griffin processes sequences with O(1) memory per step.
This lets it detect anomalies MID-SEQUENCE — before an event completes.

IMPLEMENTATION NOTES (vs full paper):
  1. Sequential RG-LRU: We use a Python for-loop over time steps.
     The paper uses a parallel scan (prefix sum) during training → O(log n) depth.
     This PoC is ~3-5x slower to train than an optimised implementation.
     At inference (beat-by-beat), the sequential loop is correct and efficient.
     Production fix: replace the for-loop with torch.cumsum-based parallel scan
     or use the reference implementation at github.com/google-deepmind/recurrentgemma.

  2. No Local Attention: The full Griffin block interleaves RG-LRU layers with
     Local Sliding Window Attention (window size 2048) at fixed intervals.
     This PoC uses only RG-LRU layers — simpler but still captures long-range
     temporal dependencies. Local attention would improve detection of sharp
     local transitions (e.g., sudden RR-interval change in ECG).
     The original griffin_ecg_poc.py in this repo does include LocalAttention.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class RGLRU(nn.Module):
    """
    Real-Gated Linear Recurrent Unit — the heart of Griffin.

    Unlike LSTM/GRU, uses a DIAGONAL recurrence matrix → can run as
    a parallel scan during training, but O(1) state during inference.

    State update:
        r_t = sigmoid(W_r @ x_t + b_r)          # recurrence gate
        i_t = sigmoid(W_i @ x_t + b_i)          # input gate
        log_a = -softplus(log(8) * r_t)          # log-space decay
        a_t = exp(log_a)                          # data-dependent decay ∈ (0,1)
        h_t = a_t * h_{t-1} + sqrt(1-a_t²) * (i_t ⊙ x_t)
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model

        # Gates — project input to gate logits
        self.W_r = nn.Linear(d_model, d_model, bias=True)
        self.W_i = nn.Linear(d_model, d_model, bias=True)
        self.W_x = nn.Linear(d_model, d_model, bias=False)

        # Init: small negative bias → decay close to 1 (long memory by default)
        nn.init.uniform_(self.W_r.bias, -4.0, -2.0)
        nn.init.zeros_(self.W_i.bias)

    def forward(self, x: torch.Tensor, h: torch.Tensor | None = None):
        """
        Args:
            x: (batch, seq_len, d_model) or (batch, d_model) for single step
            h: (batch, d_model) hidden state, None = zeros

        Returns:
            output: same shape as x
            h_last: (batch, d_model) final hidden state
        """
        single_step = x.dim() == 2
        if single_step:
            x = x.unsqueeze(1)

        B, T, D = x.shape

        if h is None:
            h = torch.zeros(B, D, device=x.device, dtype=x.dtype)

        outputs = []
        for t in range(T):
            x_t = x[:, t, :]                          # (B, D)

            r_t = torch.sigmoid(self.W_r(x_t))        # recurrence gate
            i_t = torch.sigmoid(self.W_i(x_t))        # input gate

            # Data-dependent decay: a ∈ (0, 1)
            # log(8) ≈ 2.08 — controls the range of decay values
            log_a = -F.softplus(math.log(8) * r_t)
            a_t = torch.exp(log_a)                    # decay factor

            # Normalised state update (keeps variance stable)
            gate = torch.sqrt(torch.clamp(1.0 - a_t ** 2, min=1e-6))
            h = a_t * h + gate * (i_t * self.W_x(x_t))

            outputs.append(h.unsqueeze(1))

        output = torch.cat(outputs, dim=1)            # (B, T, D)

        if single_step:
            output = output.squeeze(1)

        return output, h


class GriffinBlock(nn.Module):
    """
    One full Griffin block = Recurrent branch × Gating branch + MLP.

    Architecture:
        recurrent_out = Conv1D → RG-LRU
        gate_out      = GeLU(linear(x))
        merged        = project(recurrent_out ⊙ gate_out)
        output        = LayerNorm(x + merged)   ← residual

    Then followed by:
        output        = LayerNorm(output + MLP(output))
    """

    def __init__(self, d_model: int, expand: int = 2):
        super().__init__()
        self.d_model = d_model
        d_inner = d_model * expand

        # Pre-norm
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Recurrent branch — explicit causal padding (left-only, no future leakage)
        self.input_proj = nn.Linear(d_model, d_inner, bias=False)
        self.conv1d = nn.Conv1d(d_inner, d_inner, kernel_size=4, padding=0, groups=d_inner)
        self.rglru = RGLRU(d_inner)

        # Gating branch
        self.gate_proj = nn.Linear(d_model, d_inner, bias=False)

        # Output projection
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)

        # MLP (feed-forward)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_inner * 2),
            nn.GELU(),
            nn.Linear(d_inner * 2, d_model),
        )

    def forward(self, x: torch.Tensor, h: torch.Tensor | None = None):
        """
        Args:
            x: (batch, seq_len, d_model)
            h: hidden state from RGLRU

        Returns:
            output: (batch, seq_len, d_model)
            h_last: hidden state
        """
        residual = x
        x = self.norm1(x)

        # ── Recurrent branch ─────────────────────────────────────────────
        r = self.input_proj(x)                        # (B, T, d_inner)
        # Causal Conv1D: pad 3 zeros on the LEFT only → no future leakage
        r = r.transpose(1, 2)                         # (B, d_inner, T)
        r = F.pad(r, (3, 0))                          # explicit left-pad
        r = self.conv1d(r)                            # output: (B, d_inner, T)
        r = r.transpose(1, 2)                         # (B, T, d_inner)
        r, h_last = self.rglru(r, h)

        # ── Gating branch ────────────────────────────────────────────────
        g = F.gelu(self.gate_proj(x))

        # ── Merge ────────────────────────────────────────────────────────
        merged = self.out_proj(r * g)
        x = residual + merged

        # ── MLP (FFN) ────────────────────────────────────────────────────
        x = x + self.mlp(self.norm2(x))

        return x, h_last


class GriffinAnomalyDetector(nn.Module):
    """
    Full Griffin model for agent session anomaly detection.

    Task: given a sequence of spans (time steps), predict the next span's
    feature vector. High prediction error → anomaly.

    Input features per span (7 dims):
        0: span_type       (0-4 encoded)
        1: duration_ms     (log-normalized)
        2: tokens_used     (log-normalized)
        3: cost_cents      (log-normalized)
        4: risk_level      (0=safe, 1=warning, 2=danger)
        5: status          (0=completed, 1=error)
        6: step_index      (position within session, normalized)

    Output: reconstructed next-span features (same 7 dims)
    """

    INPUT_DIM = 7

    def __init__(self, d_model: int = 64, n_layers: int = 2):
        super().__init__()
        self.d_model = d_model

        # Project raw features → model dimension
        self.input_proj = nn.Linear(self.INPUT_DIM, d_model)

        # Stack of Griffin blocks
        self.layers = nn.ModuleList([
            GriffinBlock(d_model) for _ in range(n_layers)
        ])

        # Project back to feature space for next-step prediction
        self.output_proj = nn.Linear(d_model, self.INPUT_DIM)

        # Running baseline (exponential moving average of loss during training)
        self.register_buffer("baseline_mean", torch.tensor(0.0))
        self.register_buffer("baseline_std", torch.tensor(1.0))

    def forward(self, x: torch.Tensor, hidden_states=None):
        """
        Args:
            x: (batch, seq_len, INPUT_DIM)
            hidden_states: list of hidden states per layer, or None

        Returns:
            pred: (batch, seq_len, INPUT_DIM) — next-step predictions
            new_hidden: list of updated hidden states
        """
        h = self.input_proj(x)

        if hidden_states is None:
            hidden_states = [None] * len(self.layers)

        new_hidden = []
        for layer, hs in zip(self.layers, hidden_states):
            h, hs_new = layer(h, hs)
            new_hidden.append(hs_new)

        pred = self.output_proj(h)
        return pred, new_hidden

    def compute_anomaly_score(self, x: torch.Tensor, hidden_states=None):
        """
        Process spans one-by-one and return an anomaly score per step.
        This is the REAL-TIME inference path — O(1) memory per step.

        Returns:
            scores: list of float anomaly scores (one per span)
            hidden_states: updated hidden states
        """
        self.eval()
        scores = []

        with torch.no_grad():
            for t in range(x.size(1) - 1):
                x_t = x[:, t, :]                          # single step
                pred, hidden_states = self(x_t.unsqueeze(1), hidden_states)
                pred = pred.squeeze(1)

                # Ground truth = next span's features
                target = x[:, t + 1, :]
                mse = F.mse_loss(pred, target, reduction="none").mean(dim=-1)

                # Normalise against training baseline
                z_score = (mse - self.baseline_mean) / (self.baseline_std + 1e-8)
                scores.append(z_score.item())

        return scores, hidden_states

    def set_baseline(self, losses: list[float]):
        """Store mean/std of training losses for z-score normalisation."""
        import numpy as np
        arr = np.array(losses)
        self.baseline_mean = torch.tensor(float(arr.mean()))
        self.baseline_std = torch.tensor(float(arr.std() + 1e-8))
