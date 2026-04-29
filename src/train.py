"""
Training loop for Griffin anomaly detector.

Task: next-step prediction on span sequences.
    - Train on normal sessions ONLY → model learns "healthy" behavior
    - Anomalous sessions = held-out test set
    - At inference: high prediction error → behavioral drift detected

Usage:
    python train.py                 # synthetic data
    python train.py --real-db       # real TiDB data (needs DB_PASSWORD in data.py)
"""

import argparse
import time
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence

from griffin_model import GriffinAnomalyDetector
from data import load_data

# ── Config ────────────────────────────────────────────────────────────────────
D_MODEL    = 64
N_LAYERS   = 2
LR         = 3e-4
EPOCHS     = 30
BATCH_SIZE = 32
THRESHOLD_PERCENTILE = 95   # top 5% of training loss = anomaly threshold

DEVICE = (
    "mps"  if torch.backends.mps.is_available() else
    "cuda" if torch.cuda.is_available() else
    "cpu"
)


# ─────────────────────────────────────────────────────────────────────────────

def collate_sessions(sessions: list[list[list[float]]], device: str):
    """
    Pad a batch of sessions to the same length.
    Returns: (padded_tensor, mask)
        padded_tensor: (batch, max_len, feature_dim)
        mask: (batch, max_len) bool — True where valid
    """
    tensors = [torch.tensor(s, dtype=torch.float32) for s in sessions]
    lengths = [t.size(0) for t in tensors]
    max_len = max(lengths)

    padded = torch.zeros(len(tensors), max_len, tensors[0].size(1))
    mask   = torch.zeros(len(tensors), max_len, dtype=torch.bool)

    for i, (t, l) in enumerate(zip(tensors, lengths)):
        padded[i, :l] = t
        mask[i, :l]   = True

    return padded.to(device), mask.to(device)


def compute_loss(model, sessions_batch, device):
    """
    Next-step prediction loss on a batch of sessions.
    Only computes loss where mask is valid (ignores padding).
    """
    x, mask = collate_sessions(sessions_batch, device)

    if x.size(1) < 2:
        return None

    # Input = all steps except last, target = all steps except first
    inp    = x[:, :-1, :]      # (B, T-1, D)
    target = x[:, 1:, :]       # (B, T-1, D)
    m      = mask[:, 1:]       # (B, T-1) — valid positions

    pred, _ = model(inp)

    # MSE only at valid positions
    err = ((pred - target) ** 2).mean(dim=-1)  # (B, T-1)
    loss = (err * m.float()).sum() / m.float().sum()
    return loss


def train(model, normal_sessions, device, epochs=EPOCHS, batch_size=BATCH_SIZE):
    """Train on normal sessions. Returns training losses per epoch."""
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    all_losses = []
    model.train()

    print(f"\n{'─'*50}")
    print(f"  Training Griffin on {len(normal_sessions)} normal sessions")
    print(f"  Device: {device.upper()}  |  Epochs: {epochs}  |  d_model: {D_MODEL}")
    print(f"{'─'*50}")

    for epoch in range(1, epochs + 1):
        np.random.shuffle(normal_sessions)
        epoch_losses = []

        for start in range(0, len(normal_sessions), batch_size):
            batch = normal_sessions[start : start + batch_size]
            loss = compute_loss(model, batch, device)
            if loss is None:
                continue

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_losses.append(loss.item())

        scheduler.step()
        avg = np.mean(epoch_losses) if epoch_losses else 0.0
        all_losses.append(avg)

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{epochs}  loss={avg:.5f}  lr={scheduler.get_last_lr()[0]:.2e}")

    return all_losses


def collect_training_errors(model, normal_sessions, device):
    """
    Collect per-step prediction errors on all training sessions.
    Used to set the anomaly detection threshold.
    """
    model.eval()
    all_errors = []

    with torch.no_grad():
        for sess in normal_sessions:
            if len(sess) < 2:
                continue
            x = torch.tensor(sess, dtype=torch.float32).unsqueeze(0).to(device)
            scores, _ = model.compute_anomaly_score(x)
            all_errors.extend(scores)

    return all_errors


def evaluate(model, normal_sessions, anomalous_sessions, anomaly_indices, device, threshold):
    """
    Evaluate anomaly detection:
    - True positive:  anomalous session flagged as anomalous
    - False positive: normal session flagged as anomalous
    - Early warning:  flagged BEFORE the ground-truth anomaly index

    Returns dict of metrics.
    """
    model.eval()
    results = {
        "tp": 0, "fp": 0, "tn": 0, "fn": 0,
        "early_warnings": [],   # how many spans BEFORE anomaly we caught it
        "spans_saved": [],      # spans we skipped running = cost saved
    }

    # Evaluate normal sessions (expect no alert)
    for sess in normal_sessions[:200]:
        if len(sess) < 2:
            continue
        x = torch.tensor(sess, dtype=torch.float32).unsqueeze(0).to(device)
        scores, _ = model.compute_anomaly_score(x)
        flagged = any(s > threshold for s in scores)
        if flagged:
            results["fp"] += 1
        else:
            results["tn"] += 1

    # Evaluate anomalous sessions (expect alert)
    for i, (sess, true_at) in enumerate(zip(anomalous_sessions, anomaly_indices or [None]*len(anomalous_sessions))):
        if len(sess) < 2:
            continue
        x = torch.tensor(sess, dtype=torch.float32).unsqueeze(0).to(device)
        scores, _ = model.compute_anomaly_score(x)

        first_alert = next((j for j, s in enumerate(scores) if s > threshold), None)

        if first_alert is not None:
            results["tp"] += 1
            if true_at is not None:
                lead = true_at - first_alert
                results["early_warnings"].append(lead)
                results["spans_saved"].append(max(0, len(sess) - first_alert - 1))
        else:
            results["fn"] += 1

    n = results["tp"] + results["fp"] + results["tn"] + results["fn"]
    tp, fp, tn, fn = results["tp"], results["fp"], results["tn"], results["fn"]

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    ew = results["early_warnings"]
    ss = results["spans_saved"]

    print(f"\n{'─'*50}")
    print(f"  Evaluation Results  (threshold={threshold:.3f})")
    print(f"{'─'*50}")
    print(f"  Precision : {precision:.1%}")
    print(f"  Recall    : {recall:.1%}")
    print(f"  F1        : {f1:.1%}")
    print(f"  FP rate   : {fp/(fp+tn+1e-8):.1%}")
    if ew:
        print(f"  Avg early warning : {np.mean(ew):+.1f} spans before true anomaly")
        print(f"  Avg spans saved   : {np.mean(ss):.1f} spans per session")
    print(f"{'─'*50}\n")

    return {
        "precision": precision, "recall": recall, "f1": f1,
        "early_warnings": ew, "spans_saved": ss,
        "threshold": threshold,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--real-db", action="store_true", help="Use real TiDB data")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--save", type=str, default="griffin_model.pt")
    args = parser.parse_args()

    # ── Load data ─────────────────────────────────────────────────────────────
    normal, anomalous, anomaly_indices = load_data(use_real_db=args.real_db)

    # Train/test split on normal sessions (80/20)
    split = int(len(normal) * 0.8)
    train_sessions = normal[:split]
    test_normal    = normal[split:]

    print(f"\n  Train: {len(train_sessions)} normal sessions")
    print(f"  Test:  {len(test_normal)} normal + {len(anomalous)} anomalous")

    # ── Build model ───────────────────────────────────────────────────────────
    model = GriffinAnomalyDetector(d_model=D_MODEL, n_layers=N_LAYERS).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

    # ── Train ─────────────────────────────────────────────────────────────────
    t0 = time.time()
    train(model, train_sessions, DEVICE, epochs=args.epochs)
    print(f"\n  Training time: {time.time()-t0:.1f}s")

    # ── Set anomaly threshold ─────────────────────────────────────────────────
    print("\n  Computing anomaly threshold on training data...")
    train_errors = collect_training_errors(model, train_sessions, DEVICE)
    threshold = float(np.percentile(train_errors, THRESHOLD_PERCENTILE))
    model.set_baseline(train_errors)
    print(f"  Threshold (p{THRESHOLD_PERCENTILE}): {threshold:.4f}")

    # ── Evaluate ─────────────────────────────────────────────────────────────
    metrics = evaluate(model, test_normal, anomalous, anomaly_indices, DEVICE, threshold)

    # ── Save ─────────────────────────────────────────────────────────────────
    save_path = os.path.join(os.path.dirname(__file__), args.save)
    torch.save({
        "model_state": model.state_dict(),
        "threshold": threshold,
        "metrics": metrics,
        "config": {"d_model": D_MODEL, "n_layers": N_LAYERS},
    }, save_path)
    print(f"  Saved → {save_path}")


if __name__ == "__main__":
    main()
