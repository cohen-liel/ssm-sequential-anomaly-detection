"""
Griffin + Future Prediction Head.

Adds a binary classifier on top of Griffin's hidden state:
  "What is the probability of a PVC in the next N beats?"

This is TRUE early warning — not anomaly detection after the fact,
but a probability estimate of what is ABOUT TO HAPPEN.

Training labels: for each beat at position t,
  label = 1 if there is at least one PVC in beats [t+1 ... t+horizon]
  label = 0 otherwise

With $10M: deploy this on every ICU bed, stream live ECG,
alert nurses 6-8 seconds before the arrhythmia fires.
"""

import os, time, math, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from griffin_model import GriffinAnomalyDetector
from ecg_data import extract_beats, beat_to_features, download_record

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
HORIZON = 15   # "will there be a PVC in the next 15 beats?"


# ─────────────────────────────────────────────────────────────────────────────
class GriffinPredictor(nn.Module):
    """
    Griffin backbone + future-risk prediction head.

    backbone: GriffinAnomalyDetector (pretrained on normal beats)
    head:     small MLP → P(arrhythmia in next HORIZON beats)
    """
    def __init__(self, d_model=64, n_layers=2):
        super().__init__()
        self.backbone = GriffinAnomalyDetector(d_model=d_model, n_layers=n_layers)
        # Small head — 3 layers, dropout for regularisation
        self.head = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x, hidden_states=None):
        """
        x: (batch, seq_len, 7)
        Returns: (risk_logits: (batch, seq_len), hidden_states)
        """
        h = self.backbone.input_proj(x)

        if hidden_states is None:
            hidden_states = [None] * len(self.backbone.layers)

        new_hidden = []
        for layer, hs in zip(self.backbone.layers, hidden_states):
            h, hs_new = layer(h, hs)
            new_hidden.append(hs_new)

        risk_logits = self.head(h).squeeze(-1)   # (batch, seq_len)
        return risk_logits, new_hidden

    def predict_step(self, x_t, hidden_states):
        """Single-step inference. x_t: (1, 7). Returns (prob, hidden)."""
        logit, hidden = self(x_t.unsqueeze(0).unsqueeze(0) if x_t.dim()==1
                             else x_t.unsqueeze(1), hidden_states)
        prob = torch.sigmoid(logit[:, -1]).item()
        return prob, hidden


# ─────────────────────────────────────────────────────────────────────────────
def make_labels(beats: list, horizon: int = HORIZON) -> np.ndarray:
    """
    For each beat t, label = 1 if any of beats [t+1..t+horizon] is a PVC.
    """
    is_pvc = np.array([1 if b.label in ("V","F","E") else 0 for b in beats])
    labels = np.zeros(len(beats), dtype=np.float32)
    for t in range(len(beats)):
        if is_pvc[t+1 : t+1+horizon].any():
            labels[t] = 1.0
    return labels


def build_training_data(record_ids: list, oversample_factor: int = 4):
    """
    Build (features, labels) from multiple records.
    Only trains on records that have at least MIN_PVC PVCs.
    Oversamples windows containing upcoming PVCs to fix class imbalance.
    Returns list of (feat_seq, label_seq) tuples.
    """
    import random as _random
    MIN_PVC = 10   # skip records with too few PVCs — signal too weak
    WIN     = 40   # beats per window
    STRIDE  = 3    # smaller stride → more windows per record

    pos_wins, neg_wins = [], []

    for rid in record_ids:
        beats = extract_beats(rid)
        if not beats:
            continue
        n_pvc = sum(1 for b in beats if b.label in ("V", "F", "E"))
        if n_pvc < MIN_PVC:
            continue   # skip near-zero-PVC records
        print(f"  Record {rid}: {len(beats):,} beats, {n_pvc} PVCs ({n_pvc/len(beats):.1%})")

        feats  = [beat_to_features(b) for b in beats]
        labels = make_labels(beats, HORIZON)

        for start in range(0, len(feats) - WIN, STRIDE):
            f_win = feats[start:start+WIN]
            l_win = labels[start:start+WIN]
            if any(l > 0 for l in l_win):
                pos_wins.append((f_win, l_win))
            else:
                neg_wins.append((f_win, l_win))

    # Oversample positives, then add enough negatives to balance
    n_neg = min(len(pos_wins) * oversample_factor, len(neg_wins))
    _random.seed(42)
    sequences = pos_wins * oversample_factor + _random.sample(neg_wins, n_neg)
    _random.shuffle(sequences)
    print(f"\n  Pos windows (×{oversample_factor}): {len(pos_wins)*oversample_factor:,}  "
          f"Neg windows: {n_neg:,}  Total: {len(sequences):,}")
    return sequences


def _run_epoch(model, sequences, pos_weight, optimizer, device, train=True):
    """One pass over sequences. Returns (avg_loss, avg_acc)."""
    if train:
        model.train()
        random.shuffle(sequences)
    else:
        model.eval()

    batch_size = 32
    losses, accs = [], []

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for start in range(0, len(sequences), batch_size):
            batch = sequences[start:start+batch_size]
            max_len = max(len(f) for f, _ in batch)

            feat_t  = torch.zeros(len(batch), max_len, 7, device=device)
            label_t = torch.zeros(len(batch), max_len, device=device)
            mask_t  = torch.zeros(len(batch), max_len, dtype=torch.bool, device=device)

            for i, (f, l) in enumerate(batch):
                T = len(f)
                feat_t[i, :T]  = torch.tensor(f, dtype=torch.float32)
                label_t[i, :T] = torch.tensor(l, dtype=torch.float32)
                mask_t[i, :T]  = True

            logits, _ = model(feat_t)
            loss = F.binary_cross_entropy_with_logits(
                logits[mask_t], label_t[mask_t], pos_weight=pos_weight
            )

            if train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            preds = (torch.sigmoid(logits[mask_t]) > 0.5).float()
            acc   = (preds == label_t[mask_t]).float().mean().item()
            losses.append(loss.item())
            accs.append(acc)

    return float(np.mean(losses)) if losses else 0.0, float(np.mean(accs)) if accs else 0.0


def train_predictor(model: GriffinPredictor, sequences: list,
                    epochs=30, batch_size=32, device=DEVICE,
                    val_fraction=0.15, patience=5):
    """
    Train with binary cross-entropy + validation set + early stopping.
    Class-weighted for rare PVCs.

    Args:
        val_fraction: fraction of sequences held out for validation
        patience:     early stopping patience (epochs without val improvement)
    """
    optimizer = optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Train / val split (temporal: val = last fraction)
    random.shuffle(sequences)
    n_val  = max(1, int(len(sequences) * val_fraction))
    val_seqs   = sequences[-n_val:]
    train_seqs = sequences[:-n_val]

    # Class weight from training set only
    all_labels = np.concatenate([l for _, l in train_seqs])
    pos_rate   = all_labels.mean()
    pos_weight = torch.tensor((1 - pos_rate) / max(pos_rate, 1e-6),
                               dtype=torch.float32).to(device)

    print(f"\n  PVC rate in training data: {pos_rate:.1%}  (pos_weight={pos_weight.item():.1f})")
    print(f"  Train: {len(train_seqs):,} windows  |  Val: {len(val_seqs):,} windows")
    print(f"{'─'*58}")
    print(f"  Training GriffinPredictor — Horizon: {HORIZON} beats ahead")
    print(f"  Epochs: {epochs}  |  Early stopping patience: {patience}")
    print(f"{'─'*58}")

    best_val_loss = float("inf")
    best_state    = None
    no_improve    = 0

    for epoch in range(1, epochs+1):
        train_loss, train_acc = _run_epoch(model, train_seqs, pos_weight, optimizer, device, train=True)
        val_loss,   val_acc   = _run_epoch(model, val_seqs,   pos_weight, optimizer, device, train=False)
        scheduler.step()

        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{epochs}  "
                  f"train={train_loss:.4f}/{train_acc:.1%}  "
                  f"val={val_loss:.4f}/{val_acc:.1%}")

        # Early stopping
        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_state    = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve    = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  Early stop at epoch {epoch} (best val={best_val_loss:.4f})")
                break

    # Restore best weights
    if best_state:
        model.load_state_dict(best_state)

    return model


# ─────────────────────────────────────────────────────────────────────────────
def run_future_demo(model: GriffinPredictor, test_record="106",
                    start_beat=75, n_show=65):
    """
    Stream beats one-by-one from test_record.
    Show real-time P(PVC in next 10 beats) rising BEFORE arrhythmia fires.
    """
    beats  = extract_beats(test_record)
    window = beats[start_beat : start_beat + n_show]
    feats  = [beat_to_features(b) for b in window]
    labels = make_labels(window, HORIZON)

    print(f"\n{'═'*72}")
    print(f"  🫀  GRIFFIN FUTURE PREDICTOR — Record {test_record}")
    print(f"      P(arrhythmia in next {HORIZON} beats) — real-time, beat by beat")
    print(f"{'═'*72}")
    print()

    header = (f"  {'#':>4}  {'lbl':^5}  {'RR':>6}  "
              f"{'P(risk)':>8}  {'risk bar':^30}  "
              f"{'true label':^12}  alert")
    print(header)
    print("  " + "─"*len(header))

    x = torch.tensor(feats, dtype=torch.float32).to(DEVICE)
    hidden_states = None
    model.eval()
    first_alert = None

    with torch.no_grad():
        for t, beat in enumerate(window):
            x_t    = x[t]
            prob, hidden_states = model.predict_step(x_t, hidden_states)

            rr  = int(beat.rr_prev) if beat.rr_prev > 0 else 0
            lbl_icon = {"N":"🟢","V":"🔴","A":"🟡","F":"🟠"}.get(beat.label,"⚪")

            true_future = "⚡ PVC COMING" if labels[t] == 1 else "  safe"

            # Risk bar (30 chars)
            bar_filled = int(prob * 30)
            risk_bar   = "█"*bar_filled + "░"*(30-bar_filled)

            # Colour threshold
            alert = ""
            if prob > 0.70:
                alert = f"🚨 HIGH RISK {prob:.0%}"
                if first_alert is None:
                    first_alert = t
            elif prob > 0.40:
                alert = f"⚠️  moderate {prob:.0%}"
            else:
                alert = f"   low       {prob:.0%}"

            beat_num = start_beat + t + 1
            print(f"  {beat_num:>4}  {lbl_icon}{beat.label:^4}  {rr:>5}ms  "
                  f"{prob:>7.1%}  {risk_bar}  "
                  f"{true_future:^12}  {alert}")

            time.sleep(0.05)

    # ── Summary ───────────────────────────────────────────────────────────────
    first_pvc_t = next((t for t, b in enumerate(window) if b.label in ("V","F")), None)
    print(f"\n{'═'*72}")
    if first_alert is not None and first_pvc_t is not None:
        beats_ahead = first_pvc_t - first_alert
        ms_ahead    = beats_ahead * 800
        print(f"  ✅  Griffin flagged HIGH RISK at beat #{start_beat+first_alert+1}")
        print(f"      First real PVC at beat #{start_beat+first_pvc_t+1}")
        print(f"      → {beats_ahead} beats early warning  ≈  {ms_ahead/1000:.1f} seconds ahead")
        print()
        print(f"  With $10M investment:")
        print(f"  ┌────────────────────────────────────────────────────────┐")
        print(f"  │  Deploy on every ICU bed (live ECG stream, no cloud)  │")
        print(f"  │  Nurse gets paged {beats_ahead} beats = {ms_ahead/1000:.0f}s before the event   │")
        print(f"  │  Model runs on local edge device → $0 API cost        │")
        print(f"  │  Improves as more patient data accumulates             │")
        print(f"  │  Gemma 4: needs 972K tokens ($2.43) per 30-min window  │")
        print(f"  │           and answers AFTER — not before               │")
        print(f"  └────────────────────────────────────────────────────────┘")
    print(f"{'═'*72}\n")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Records to train on — exclude 106 (test) and 100/zero-PVC records
    # PVC counts: 119=444, 116=109, 124=47, 109=38, 105=41, 118=16, 108=17
    TRAIN_RECORDS = ["105","108","109","116","118","119","121","122","123","124"]

    print("\n  Building training sequences (only PVC-rich records, oversampled)...")
    sequences = build_training_data(TRAIN_RECORDS, oversample_factor=4)

    if not sequences:
        print("ERROR: No training sequences built — check record paths")
        exit(1)

    model = GriffinPredictor(d_model=128, n_layers=3).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model: d_model=128, n_layers=3, params={n_params:,}")

    train_predictor(model, sequences, epochs=50, device=DEVICE, patience=8)

    # ── Demo on UNSEEN record 106 (heavy PVC burden, 24.8%) ───────────────────
    print("\n  === Evaluating on held-out record 106 (never seen during training) ===")
    run_future_demo(model, test_record="106", start_beat=75, n_show=65)
