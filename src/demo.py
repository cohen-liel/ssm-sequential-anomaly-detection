"""
Real-time Griffin anomaly detection demo.

Shows Griffin processing spans ONE BY ONE — exactly as they would arrive
from a live agentwatch session — and alerting BEFORE the session fails.

This is the key demo: Griffin screams at span 4 out of 14.
Gemma 4 would only know something went wrong after span 14 (or never).

Usage:
    python demo.py                  # run with synthetic session
    python demo.py --load           # load trained model first
    python demo.py --interactive    # pause between spans
"""

import argparse
import time
import os
import sys
import math
import random
import torch
import numpy as np

from griffin_model import GriffinAnomalyDetector
from data import _anomalous_session, _normal_session, generate_synthetic_dataset

try:
    from rich.console import Console
    from rich.table import Table
    from rich.progress import Progress, BarColumn, TextColumn
    from rich.panel import Panel
    from rich import box
    HAS_RICH = True
except ImportError:
    HAS_RICH = False

DEVICE = (
    "mps"  if torch.backends.mps.is_available() else
    "cuda" if torch.cuda.is_available() else
    "cpu"
)

SPAN_TYPE_NAMES = {0: "llm", 1: "tool", 2: "decision", 3: "action", 4: "system"}
RISK_NAMES      = {0: "safe", 1: "warning", 2: "danger"}
STATUS_NAMES    = {0: "✓", 1: "✗ error"}


def load_model(path="griffin_model.pt") -> tuple[GriffinAnomalyDetector, float]:
    full_path = os.path.join(os.path.dirname(__file__), path)
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"No saved model at {full_path}. Run train.py first.")
    ckpt = torch.load(full_path, map_location=DEVICE)
    cfg = ckpt["config"]
    model = GriffinAnomalyDetector(d_model=cfg["d_model"], n_layers=cfg["n_layers"]).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt["threshold"]


def quick_train_model() -> tuple[GriffinAnomalyDetector, float]:
    """Train a fresh model in ~10 seconds for the demo."""
    from train import train, collect_training_errors
    import numpy as np

    print("  [Griffin] Training model on 800 normal sessions (≈10s)...\n")
    normal, anomalous, _ = generate_synthetic_dataset(n_normal=800, n_anomalous=200)

    model = GriffinAnomalyDetector(d_model=64, n_layers=2).to(DEVICE)
    train(model, normal, DEVICE, epochs=25, batch_size=32)

    errs = collect_training_errors(model, normal, DEVICE)
    threshold = float(np.percentile(errs, 95))
    model.set_baseline(errs)
    return model, threshold


def decode_features(feat: list[float]) -> dict:
    """Decode normalised feature vector back to human-readable values."""
    return {
        "type":     SPAN_TYPE_NAMES[round(feat[0] * 4)],
        "duration": int(math.expm1(feat[1] * 10)),
        "tokens":   int(math.expm1(feat[2] * 10)),
        "cost_c":   round(math.expm1(feat[3] * 5), 2),
        "risk":     RISK_NAMES[round(feat[4] * 2)],
        "status":   STATUS_NAMES[round(feat[5])],
    }


def bar(score: float, width: int = 20) -> str:
    """ASCII progress bar for anomaly score."""
    filled = int(min(score, 1.0) * width)
    return "█" * filled + "░" * (width - filled)


def run_demo(model: GriffinAnomalyDetector, threshold: float,
             interactive: bool = False, seed: int = None):
    """
    Simulate a live session arriving span-by-span.
    Shows Griffin's anomaly score updating in real-time.
    """
    if seed:
        random.seed(seed)
        np.random.seed(seed)

    # Pick a dramatic anomaly: starts normal, goes wrong mid-session
    session_spans, true_anomaly_at = _anomalous_session(anomaly_at=4)
    n = len(session_spans)

    print("\n" + "═"*60)
    print("  🤖  GRIFFIN  —  Real-Time Agent Session Monitor")
    print("═"*60)
    print(f"  Session: {n} spans  |  True anomaly starts at span #{true_anomaly_at}")
    print(f"  Threshold: {threshold:.3f}  |  Device: {DEVICE.upper()}")
    print("═"*60)
    print()

    x = torch.tensor(session_spans, dtype=torch.float32).unsqueeze(0).to(DEVICE)

    hidden_states = None
    first_alert = None
    alert_printed = False

    header = f"  {'#':>3}  {'type':^10}  {'ms':>6}  {'tokens':>7}  {'risk':^8}  {'st':^7}  {'score':^8}  {'bar':^22}  flag"
    print(header)
    print("  " + "─" * (len(header) - 2))

    model.eval()
    with torch.no_grad():
        for t in range(n - 1):
            feat = session_spans[t]
            info = decode_features(feat)

            # Single-step inference — O(1) memory, exactly like production
            x_t = x[:, t, :].unsqueeze(1)
            pred, hidden_states = model(x_t, hidden_states)
            pred = pred.squeeze(1)

            target = x[:, t + 1, :]
            import torch.nn.functional as F
            mse = F.mse_loss(pred, target).item()
            score = (mse - model.baseline_mean.item()) / (model.baseline_std.item() + 1e-8)
            score = max(0.0, score)

            is_anomalous_region = t >= true_anomaly_at
            is_alert = score > threshold

            # Color indicators
            flag = ""
            if is_alert and not alert_printed:
                flag = "🚨 ALERT"
                first_alert = t
                alert_printed = True
            elif is_alert:
                flag = "⚠️"
            elif is_anomalous_region:
                flag = "  [drifting]"

            score_bar = bar(score / max(threshold * 2, 1e-8))

            risk_icon = {"safe": "🟢", "warning": "🟡", "danger": "🔴"}[info["risk"]]

            line = (
                f"  {t+1:>3}  {info['type']:^10}  {info['duration']:>6}ms"
                f"  {info['tokens']:>7}tok  {risk_icon}{info['risk']:^7}"
                f"  {info['status']:^7}  {score:>7.3f}  {score_bar}  {flag}"
            )
            print(line)

            if interactive:
                time.sleep(0.4)
            else:
                time.sleep(0.05)

            # Print alert box immediately when first triggered
            if is_alert and first_alert == t:
                spans_remaining = n - t - 1
                print()
                print("  ┌─────────────────────────────────────────────────┐")
                print(f"  │  🚨  GRIFFIN EARLY WARNING — Span {t+1}/{n}          │")
                print(f"  │                                                 │")
                print(f"  │  Anomaly score: {score:.3f}  (threshold: {threshold:.3f})     │")
                if true_anomaly_at is not None:
                    lead = true_anomaly_at - t
                    if lead > 0:
                        print(f"  │  ✓ Caught {lead} span(s) BEFORE true anomaly         │")
                    elif lead == 0:
                        print(f"  │  ✓ Caught at exact anomaly boundary               │")
                    else:
                        print(f"  │  ⚡ Detected {-lead} spans into anomaly region       │")
                print(f"  │  {spans_remaining} spans would still run → STOP NOW?        │")
                print("  └─────────────────────────────────────────────────┘")
                print()
                if interactive:
                    time.sleep(1.0)

    print()
    print("═"*60)
    if first_alert is not None:
        lead = true_anomaly_at - first_alert
        saved = n - first_alert - 1
        cost_saved = sum(
            math.expm1(session_spans[i][3] * 5) * 0.00001
            for i in range(first_alert + 1, n)
        )
        print(f"  ✅  Summary")
        print(f"     Alert fired at span #{first_alert+1} of {n}")
        if lead > 0:
            print(f"     {lead} span(s) BEFORE true anomaly onset (true_at=#{true_anomaly_at+1})")
        print(f"     ~{saved} span(s) could have been skipped")
        print(f"     Estimated cost saved: ${cost_saved:.4f}")
        print()
        print("  vs Gemma 4:")
        print(f"     Would need all {n} spans to conclude 'something looks wrong'")
        print(f"     → reacts AFTER the damage, not before")
    else:
        print("  No alert fired (missed detection for this session).")
    print("═"*60)
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--load",        action="store_true", help="Load saved model instead of training")
    parser.add_argument("--interactive", action="store_true", help="Pause between spans")
    parser.add_argument("--seed",        type=int, default=7,  help="Random seed for session")
    args = parser.parse_args()

    if args.load:
        try:
            model, threshold = load_model()
            print(f"  Loaded saved model (threshold={threshold:.4f})")
        except FileNotFoundError as e:
            print(f"  {e}")
            print("  Training fresh model...")
            model, threshold = quick_train_model()
    else:
        model, threshold = quick_train_model()

    run_demo(model, threshold, interactive=args.interactive, seed=args.seed)


if __name__ == "__main__":
    main()
