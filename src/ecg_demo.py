"""
ECG Arrhythmia Demo — Griffin on real MIT-BIH data.

Downloads real ECG records from PhysioNet, trains Griffin on normal sinus
rhythm, then watches a new record beat-by-beat and alerts on arrhythmias.

Usage:
    python ecg_demo.py              # full demo (downloads ~20MB)
    python ecg_demo.py --tokens     # just show token comparison
"""

import argparse
import os
import time
import math
import random
import numpy as np
import torch
import torch.nn.functional as F

from griffin_model import GriffinAnomalyDetector
from ecg_data import load_mitbih, token_comparison, beat_to_features, extract_beats, download_record

DEVICE = (
    "mps"  if torch.backends.mps.is_available() else
    "cuda" if torch.cuda.is_available() else
    "cpu"
)

D_MODEL  = 64
N_LAYERS = 2


def split_record_into_windows(features: list, window: int = 30, stride: int = 5):
    """
    Split a long record (2000 beats) into overlapping windows of 30 beats.
    Each window = one training sample. Griffin trains on sequences of beats.
    """
    seqs = []
    for start in range(0, len(features) - window, stride):
        seqs.append(features[start : start + window])
    return seqs


def train_on_ecg(normal_records: list, epochs: int = 20) -> tuple[GriffinAnomalyDetector, float, list]:
    """Train Griffin on normal sinus rhythm windows. Return model + threshold."""
    from train import train as griffin_train, collect_training_errors

    # Flatten all normal records into 30-beat windows
    all_windows = []
    for rec in normal_records:
        all_windows.extend(split_record_into_windows(rec, window=30, stride=5))

    print(f"\n  Training on {len(all_windows):,} normal beat windows "
          f"from {len(normal_records)} records...")

    model = GriffinAnomalyDetector(d_model=D_MODEL, n_layers=N_LAYERS).to(DEVICE)
    griffin_train(model, all_windows, DEVICE, epochs=epochs, batch_size=64)

    errors = collect_training_errors(model, all_windows[:500], DEVICE)
    threshold = float(np.percentile(errors, 95))
    model.set_baseline(errors)
    print(f"  Threshold (p95): {threshold:.4f}")

    return model, threshold, errors


def run_ecg_demo(model: GriffinAnomalyDetector, threshold: float, test_record_id: str = "106"):
    """
    Stream beats from a known arrhythmia-heavy record one-by-one.
    Record 106 has many PVCs (premature ventricular contractions).
    """
    import wfdb

    DATA_DIR = os.path.join(os.path.dirname(__file__), "mit_bih_data")
    download_record(test_record_id)

    from ecg_data import extract_beats
    beats = extract_beats(test_record_id)

    if not beats:
        print(f"  Could not load record {test_record_id}")
        return

    n_total   = len(beats)
    n_normal  = sum(1 for b in beats if b.is_normal)
    n_arrhyth = n_total - n_normal

    print(f"\n{'═'*62}")
    print(f"  🫀  GRIFFIN ECG MONITOR — Record {test_record_id} (MIT-BIH)")
    print(f"{'═'*62}")
    print(f"  Total beats:     {n_total:,}")
    print(f"  Normal (N):      {n_normal:,}  ({n_normal/n_total:.1%})")
    print(f"  Arrhythmia:      {n_arrhyth:,}  ({n_arrhyth/n_total:.1%})")
    print(f"  Threshold:       {threshold:.4f}")
    print(f"{'═'*62}")
    print()

    # Stream first 80 beats to show the transition
    demo_beats = beats[:80]
    features   = [beat_to_features(b) for b in demo_beats]

    x = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(DEVICE)

    hidden_states = None
    alerts = 0

    header = f"  {'#':>4}  {'label':^6}  {'RR(ms)':>7}  {'amp':>6}  {'score':>8}  {'bar':^22}  result"
    print(header)
    print("  " + "─" * (len(header) - 2))

    model.eval()
    with torch.no_grad():
        for t in range(len(demo_beats) - 1):
            beat = demo_beats[t]
            feat = features[t]

            x_t  = x[:, t, :].unsqueeze(1)
            pred, hidden_states = model(x_t, hidden_states)
            pred = pred.squeeze(1)

            target = x[:, t + 1, :]
            mse    = F.mse_loss(pred, target).item()
            score  = max(0.0, (mse - model.baseline_mean.item()) / (model.baseline_std.item() + 1e-8))

            is_alert = score > threshold
            true_anom = not beat.is_normal

            rr_ms = int(beat.rr_prev) if beat.rr_prev > 0 else 0
            amp   = round(beat.amplitude, 2)

            if is_alert:
                alerts += 1
                flag = "🚨 ARRHYTHMIA" if true_anom else "⚠️  FP"
            elif true_anom:
                flag = "  [missed]"
            else:
                flag = "  ✓"

            score_display = min(score / max(threshold * 2, 1e-8), 1.0)
            filled = int(score_display * 22)
            bar_str = "█" * filled + "░" * (22 - filled)

            label_icon = {"N":"🟢","V":"🔴","A":"🟡","F":"🟠","E":"🔴"}.get(beat.label, "⚪")

            print(f"  {t+1:>4}  {label_icon}{beat.label:^5}  {rr_ms:>7}ms  {amp:>6.3f}  "
                  f"{score:>8.3f}  {bar_str}  {flag}")

            time.sleep(0.03)

    # Summary
    true_pos  = sum(1 for i, b in enumerate(demo_beats[:-1])
                    if not b.is_normal)
    print(f"\n{'═'*62}")
    print(f"  📊 Results on record {test_record_id} (first 80 beats)")
    print(f"     Alerts fired:  {alerts}")
    print(f"     True arrhythmias in window: {sum(1 for b in demo_beats if not b.is_normal)}")
    print()
    print(f"  vs GPT-4o / Gemma 4 analyzing same record:")
    samples = 360 * 60 * 30
    tokens  = int(samples * 1.5)
    print(f"     Would need {tokens:,} tokens for 30-min record")
    print(f"     Cost: ${tokens/1e6 * 2.50:.4f}  per analysis")
    print(f"     Detects AFTER full record ingested (30 min lag!)")
    print()
    print(f"  Griffin:")
    print(f"     Alerts in real-time, beat by beat")
    print(f"     O(1) memory, $0.00 API cost, no context limit")
    print(f"{'═'*62}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", action="store_true", help="Only show token comparison")
    parser.add_argument("--records", type=int, default=8, help="Num records to train on")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--test-record", type=str, default="106",
                        help="MIT-BIH record to test on (106=many PVCs)")
    args = parser.parse_args()

    if args.tokens:
        token_comparison()
        return

    # ── Load data ─────────────────────────────────────────────────────────────
    normal_records, anomalous_records = load_mitbih(n_records=args.records)

    if not normal_records:
        print("  No normal records loaded. Check network connection.")
        return

    # ── Token comparison (show BEFORE training so user understands the stakes)
    token_comparison()

    # ── Train ─────────────────────────────────────────────────────────────────
    model, threshold, _ = train_on_ecg(normal_records, epochs=args.epochs)

    # ── Demo: real-time ECG monitoring ────────────────────────────────────────
    run_ecg_demo(model, threshold, test_record_id=args.test_record)


if __name__ == "__main__":
    main()
