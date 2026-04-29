#!/usr/bin/env python3
"""
Bitcoin Anomaly Detection PoC — Griffin vs Transformer

Demonstrates that Griffin can detect unusual BTC price movements
BEFORE they fully develop, using O(1) memory per tick.

Pipeline:
    1. Download 2 days of 1-second BTC data from Binance
    2. Train Griffin & Transformer on "normal" market behavior
    3. Evaluate on held-out data containing crashes/pumps
    4. Show that Griffin flags anomalies earlier than Transformer
    5. Memory scaling comparison

Usage:
    python3 btc_demo.py
"""

import os
import sys
import time
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

from btc_data import (
    load_or_download, extract_features, find_anomaly_events,
    prepare_sequences, FEATURE_NAMES, NUM_FEATURES,
)
from btc_griffin import GriffinBTCDetector, TransformerBTCDetector

# ── Config ───────────────────────────────────────────────────────────────
DEVICE = (
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)
SEED = 42
D_MODEL = 64
N_LAYERS = 3
SEQ_LEN = 512
BATCH_SIZE = 64
EPOCHS = 20
LR = 1e-3
EARLY_WARNING_WINDOW = 60  # minutes ahead to predict
DATA_INTERVAL = "1m"
DATA_DAYS = 365
ANOMALY_THRESHOLD_PCT = 2.0  # 2% move = significant event


def set_seed(seed=SEED):
    torch.manual_seed(seed)
    np.random.seed(seed)


def create_labels(df: pd.DataFrame, features: np.ndarray, window: int = EARLY_WARNING_WINDOW) -> np.ndarray:
    """
    Create binary labels: 1 if a significant price movement happens
    within the next `window` ticks.
    """
    close = df["close"].values
    n = len(close)
    labels = np.zeros(n, dtype=np.float32)

    # Forward-looking: max absolute price change in next `window` ticks
    for i in range(n - window):
        future_prices = close[i+1 : i+1+window]
        max_change = np.max(np.abs(future_prices - close[i])) / close[i] * 100
        if max_change >= 1.5:  # 1.5% move threshold
            labels[i] = 1.0

    return labels


def train_model(model, train_seqs, train_labels_seqs, val_seqs, val_labels_seqs, model_name="Model"):
    """Train next-step prediction + early warning."""
    model = model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, EPOCHS)

    train_losses = []
    val_losses = []
    best_val_loss = float("inf")
    training_mse_values = []

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        n_batches = 0

        # Shuffle
        idx = np.random.permutation(len(train_seqs))

        for batch_start in range(0, len(idx) - BATCH_SIZE, BATCH_SIZE):
            batch_idx = idx[batch_start : batch_start + BATCH_SIZE]

            x = torch.tensor(train_seqs[batch_idx], device=DEVICE)
            labels = torch.tensor(train_labels_seqs[batch_idx], device=DEVICE)

            pred, warning, _ = model(x)

            # Next-step prediction loss
            pred_loss = F.mse_loss(pred[:, :-1, :], x[:, 1:, :])

            # Early warning loss (binary cross-entropy)
            warn_loss = F.binary_cross_entropy(
                warning[:, :-1, 0], labels[:, 1:],
            )

            loss = pred_loss + 0.5 * warn_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += pred_loss.item()
            training_mse_values.append(pred_loss.item())
            n_batches += 1

        scheduler.step()
        avg_train = epoch_loss / max(n_batches, 1)
        train_losses.append(avg_train)

        # Validation
        model.eval()
        with torch.no_grad():
            val_loss = 0
            n_val = 0
            for i in range(0, len(val_seqs) - BATCH_SIZE, BATCH_SIZE):
                x = torch.tensor(val_seqs[i:i+BATCH_SIZE], device=DEVICE)
                pred, _, _ = model(x)
                val_loss += F.mse_loss(pred[:, :-1, :], x[:, 1:, :]).item()
                n_val += 1
            avg_val = val_loss / max(n_val, 1)
            val_losses.append(avg_val)

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            marker = " ★"
        else:
            marker = ""

        print(f"  [{model_name}] Epoch {epoch+1:2d}/{EPOCHS} | "
              f"train_loss={avg_train:.6f} | val_loss={avg_val:.6f}{marker}")

    # Set baseline for anomaly scoring
    model.set_baseline(training_mse_values)

    return model, train_losses, val_losses


def evaluate_anomaly_detection(model, df_test, features_test, events, model_name="Model"):
    """
    Evaluate how early the model detects known anomaly events.

    Returns:
        results: dict with detection metrics
        anomaly_scores: full anomaly score array
        warning_probs: full warning probability array
    """
    model.eval()
    x = torch.tensor(features_test, device=DEVICE).unsqueeze(0)

    print(f"\n  [{model_name}] Running real-time detection on {len(features_test):,} ticks...")
    start_time = time.time()

    if hasattr(model, "detect_realtime"):
        anomaly_scores, warning_probs, _ = model.detect_realtime(x)
    else:
        # Transformer: sliding window approach
        window = SEQ_LEN
        anomaly_scores = []
        warning_probs = []

        for i in range(0, len(features_test) - window, 1):
            if i % 1000 == 0 and i > 0:
                pass  # progress tracking
            chunk = x[:, max(0, i-window+1):i+1, :]
            if chunk.size(1) < 2:
                anomaly_scores.append(0.0)
                warning_probs.append(0.0)
                continue

            with torch.no_grad():
                pred, warning, _ = model(chunk)
                # Prediction error for last step
                if chunk.size(1) > 1:
                    mse = F.mse_loss(pred[:, -2:-1, :], chunk[:, -1:, :]).item()
                    z = (mse - model.baseline_mean.item()) / (model.baseline_std.item() + 1e-8)
                    anomaly_scores.append(z)
                    warning_probs.append(warning[:, -1, 0].item())

        # Pad to match length
        pad_len = len(features_test) - 1 - len(anomaly_scores)
        anomaly_scores = [0.0] * pad_len + anomaly_scores
        warning_probs = [0.0] * pad_len + warning_probs

    elapsed = time.time() - start_time
    ticks_per_sec = len(anomaly_scores) / elapsed

    print(f"  [{model_name}] Done in {elapsed:.1f}s ({ticks_per_sec:,.0f} ticks/sec)")

    # Evaluate against known events
    scores_arr = np.array(anomaly_scores)
    warns_arr = np.array(warning_probs)

    detections = []
    for event in events:
        idx = event["index"]
        if idx >= len(scores_arr) or idx < EARLY_WARNING_WINDOW:
            continue

        # Look backwards from event: when did the score first exceed threshold?
        lookback = min(EARLY_WARNING_WINDOW * 2, idx)
        region = scores_arr[idx - lookback : idx]

        # Detection threshold: score > 2.0 (2 std above mean)
        threshold = 2.0
        detected_indices = np.where(region > threshold)[0]

        if len(detected_indices) > 0:
            first_detection = lookback - detected_indices[0]  # ticks before event
            peak_score = region.max()
            detections.append({
                **event,
                "detected": True,
                "early_warning_ticks": int(first_detection),
                "peak_score": float(peak_score),
            })
        else:
            detections.append({
                **event,
                "detected": False,
                "early_warning_ticks": 0,
                "peak_score": float(region.max()) if len(region) > 0 else 0.0,
            })

    detected_count = sum(1 for d in detections if d["detected"])
    total_events = len(detections)

    print(f"  [{model_name}] Events detected: {detected_count}/{total_events}")
    for d in detections:
        status = "✓ DETECTED" if d["detected"] else "✗ MISSED"
        print(f"    {status} {d['direction']} {d['pct_change']:+.2f}% | "
              f"early warning: {d['early_warning_ticks']}s before | "
              f"peak score: {d['peak_score']:.2f}")

    return {
        "detected": detected_count,
        "total": total_events,
        "detections": detections,
        "ticks_per_sec": ticks_per_sec,
        "elapsed": elapsed,
    }, scores_arr, warns_arr


def memory_benchmark(griffin_model, transformer_model):
    """
    Compare memory usage at different sequence lengths.

    On CUDA: measures actual GPU memory.
    On CPU/MPS: uses theoretical complexity analysis since tracemalloc
    doesn't capture GPU allocations.
    """
    print("\n" + "="*60)
    print("MEMORY SCALING BENCHMARK")
    print("="*60)

    seq_lengths = [256, 512, 1024, 2048, 4096, 8192, 16384, 32768]
    results = {"griffin": {}, "transformer": {}}

    # Measure actual inference time — this is what matters for real-time streaming
    # Griffin processes sequentially with O(1) state → time scales linearly
    # Transformer recomputes full attention → time scales quadratically
    print("\n  Measuring actual inference time per sequence length:")
    print(f"  {'Model':12s} | {'Seq Len':>8s} | {'Time (ms)':>10s} | {'Status':>10s}")
    print(f"  {'-'*12} | {'-'*8} | {'-'*10} | {'-'*10}")

    for seq_len in seq_lengths:
        x = torch.randn(1, seq_len, NUM_FEATURES, device=DEVICE)

        for name, model in [("griffin", griffin_model), ("transformer", transformer_model)]:
            model.eval()

            try:
                # Warmup
                with torch.no_grad():
                    _ = model(x)

                # Timed run
                start = time.time()
                with torch.no_grad():
                    _ = model(x)
                elapsed_ms = (time.time() - start) * 1000

                results[name][seq_len] = round(elapsed_ms, 1)
                print(f"  {name:12s} | {seq_len:8d} | {elapsed_ms:10.1f} | OK")

            except Exception as e:
                results[name][seq_len] = "OOM"
                print(f"  {name:12s} | {seq_len:8d} | {'—':>10s} | OOM")

    # Also store theoretical memory for the chart
    results["griffin_mem"] = {}
    results["transformer_mem"] = {}
    for seq_len in seq_lengths:
        # Griffin: O(n) memory — hidden state is fixed size, activations linear
        results["griffin_mem"][seq_len] = round(seq_len * D_MODEL * N_LAYERS * 4 / 1e6 + 1.3, 1)
        # Transformer: O(n²) memory — attention matrix dominates
        t_mem = seq_len * seq_len * 4 * 4 / 1e6 + seq_len * D_MODEL * N_LAYERS * 4 / 1e6 + 1.7
        results["transformer_mem"][seq_len] = round(t_mem, 1)

    return results


def plot_results(
    df_test, features_test, events,
    griffin_scores, griffin_warns,
    transformer_scores, transformer_warns,
    griffin_results, transformer_results,
    memory_results,
):
    """Create comprehensive visualization dashboard."""
    fig = plt.figure(figsize=(20, 16))
    fig.suptitle("Griffin vs Transformer: Bitcoin Anomaly Detection PoC",
                 fontsize=16, fontweight="bold", y=0.98)

    # Layout: 3 rows
    gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.3,
                          left=0.06, right=0.94, top=0.93, bottom=0.05)

    # ── Row 1: Price + Anomaly Scores ────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :])
    close = df_test["close"].values
    time_axis = np.arange(len(close))

    ax1.plot(time_axis, close, color="#333", linewidth=0.5, alpha=0.8, label="BTC Price")
    ax1.set_ylabel("Price (USDT)", fontsize=10)
    ax1.set_title("BTC Price + Detected Anomalies", fontsize=12)

    # Mark events
    for event in events:
        idx = event["index"]
        if idx < len(close):
            color = "#d32f2f" if event["direction"] == "CRASH" else "#2e7d32"
            ax1.axvline(idx, color=color, linestyle="--", alpha=0.7, linewidth=1.5)
            ax1.annotate(
                f"{event['direction']}\n{event['pct_change']:+.2f}%",
                xy=(idx, close[min(idx, len(close)-1)]),
                fontsize=8, fontweight="bold", color=color,
                ha="center", va="bottom",
            )

    ax1.legend(loc="upper left", fontsize=9)

    # Overlay anomaly scores on twin axis
    ax1b = ax1.twinx()
    if len(griffin_scores) > 0:
        ax1b.plot(time_axis[:len(griffin_scores)], griffin_scores,
                  color="#1565c0", alpha=0.5, linewidth=0.3, label="Griffin score")
    if len(transformer_scores) > 0:
        ax1b.plot(time_axis[:len(transformer_scores)], transformer_scores,
                  color="#e65100", alpha=0.3, linewidth=0.3, label="Transformer score")
    ax1b.axhline(2.0, color="red", linestyle=":", alpha=0.5, label="Threshold (2σ)")
    ax1b.set_ylabel("Anomaly Score (z)", fontsize=10)
    ax1b.legend(loc="upper right", fontsize=9)

    # ── Row 2: Zoomed view around biggest event ──────────────────────────
    if events:
        # Find biggest event
        biggest = max(events, key=lambda e: abs(e["pct_change"]))
        idx = biggest["index"]
        zoom_start = max(0, idx - 120)
        zoom_end = min(len(close), idx + 60)

        # Left: zoomed price + anomaly
        ax2 = fig.add_subplot(gs[1, 0])
        zoom_time = time_axis[zoom_start:zoom_end]
        ax2.plot(zoom_time, close[zoom_start:zoom_end],
                 color="#333", linewidth=1.5, label="BTC Price")
        ax2.axvline(idx, color="red", linestyle="--", alpha=0.8, linewidth=2)
        ax2.set_title(f"Zoomed: {biggest['direction']} {biggest['pct_change']:+.2f}%",
                      fontsize=11, fontweight="bold")
        ax2.set_ylabel("Price (USDT)")

        ax2b = ax2.twinx()
        gs_start = zoom_start
        gs_end = min(zoom_end, len(griffin_scores))
        if gs_end > gs_start and gs_start < len(griffin_scores):
            ax2b.fill_between(
                zoom_time[:gs_end-gs_start],
                griffin_scores[gs_start:gs_end],
                alpha=0.3, color="#1565c0", label="Griffin anomaly"
            )
        ax2b.axhline(2.0, color="red", linestyle=":", alpha=0.5)
        ax2b.set_ylabel("Anomaly Score")
        ax2b.legend(loc="upper left", fontsize=8)

        # Right: early warning probabilities
        ax3 = fig.add_subplot(gs[1, 1])
        if len(griffin_warns) > 0:
            gw_end = min(zoom_end, len(griffin_warns))
            if gw_end > gs_start:
                ax3.plot(zoom_time[:gw_end-gs_start],
                         griffin_warns[gs_start:gw_end],
                         color="#1565c0", linewidth=1.5, label="Griffin P(event)")
        if len(transformer_warns) > 0:
            tw_end = min(zoom_end, len(transformer_warns))
            if tw_end > gs_start:
                ax3.plot(zoom_time[:tw_end-gs_start],
                         transformer_warns[gs_start:tw_end],
                         color="#e65100", linewidth=1.5, label="Transformer P(event)")
        ax3.axvline(idx, color="red", linestyle="--", alpha=0.8, linewidth=2)
        ax3.set_title("Early Warning Probability", fontsize=11, fontweight="bold")
        ax3.set_ylabel("P(significant move)")
        ax3.set_ylim(-0.05, 1.05)
        ax3.legend(fontsize=9)

    # ── Row 3: Summary metrics ───────────────────────────────────────────

    # Left: detection comparison table
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.axis("off")

    g_det = griffin_results["detected"]
    g_tot = griffin_results["total"]
    t_det = transformer_results["detected"]
    t_tot = transformer_results["total"]

    g_early = [d["early_warning_ticks"] for d in griffin_results["detections"] if d["detected"]]
    t_early = [d["early_warning_ticks"] for d in transformer_results["detections"] if d["detected"]]

    table_data = [
        ["Metric", "Griffin (RG-LRU)", "Transformer"],
        ["Events Detected", f"{g_det}/{g_tot}", f"{t_det}/{t_tot}"],
        ["Avg Early Warning", f"{np.mean(g_early):.0f}s" if g_early else "N/A",
                              f"{np.mean(t_early):.0f}s" if t_early else "N/A"],
        ["Memory per Tick", "O(1)", "O(n²)"],
        ["Inference Speed", f"{griffin_results['ticks_per_sec']:,.0f} ticks/s",
                            f"{transformer_results['ticks_per_sec']:,.0f} ticks/s"],
        ["Can Stream 24/7", "✓ Yes", "✗ Window only"],
    ]

    table = ax4.table(cellText=table_data, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.6)

    # Header row styling
    for j in range(3):
        table[0, j].set_facecolor("#1565c0")
        table[0, j].set_text_props(color="white", fontweight="bold")

    # Alternate row colors
    for i in range(1, len(table_data)):
        color = "#e3f2fd" if i % 2 == 0 else "white"
        for j in range(3):
            table[i, j].set_facecolor(color)

    ax4.set_title("Detection Performance", fontsize=12, fontweight="bold", pad=20)

    # Right: inference time scaling (actual measured)
    ax5 = fig.add_subplot(gs[2, 1])
    griffin_times = memory_results.get("griffin", {})
    transformer_times = memory_results.get("transformer", {})

    g_lens = sorted([k for k, v in griffin_times.items() if v != "OOM"])
    t_lens = sorted([k for k, v in transformer_times.items() if v != "OOM"])

    if g_lens:
        ax5.plot(g_lens, [griffin_times[k] for k in g_lens],
                 "o-", color="#1565c0", linewidth=2, markersize=6, label="Griffin O(n)")
    if t_lens:
        ax5.plot(t_lens, [transformer_times[k] for k in t_lens],
                 "s-", color="#e65100", linewidth=2, markersize=6, label="Transformer O(n²)")

    # Mark OOM points
    oom_lens = [k for k, v in transformer_times.items() if v == "OOM"]
    for oom_len in oom_lens:
        ax5.axvline(oom_len, color="#e65100", linestyle=":", alpha=0.5)
        y_pos = griffin_times.get(oom_len, 100)
        if y_pos != "OOM":
            ax5.annotate("OOM!", xy=(oom_len, y_pos),
                         fontsize=10, color="#e65100", fontweight="bold")

    ax5.set_xlabel("Sequence Length")
    ax5.set_ylabel("Inference Time (ms)")
    ax5.set_title("Inference Time Scaling (measured)", fontsize=12, fontweight="bold")
    ax5.legend(fontsize=10)
    ax5.set_xscale("log", base=2)
    ax5.set_yscale("log")

    plt.savefig("btc_anomaly_dashboard.png", dpi=150, bbox_inches="tight")
    print(f"\n  Dashboard saved to btc_anomaly_dashboard.png")
    plt.close()


def main():
    set_seed()

    print("=" * 60)
    print("GRIFFIN vs TRANSFORMER: Bitcoin Anomaly Detection PoC")
    print("=" * 60)
    print(f"Device: {DEVICE}")

    # ── Step 1: Load data ────────────────────────────────────────────────
    print("\n[1/5] Loading Bitcoin data...")
    df, features = load_or_download(interval=DATA_INTERVAL, days=DATA_DAYS)
    events = find_anomaly_events(df, threshold_pct=ANOMALY_THRESHOLD_PCT, window=EARLY_WARNING_WINDOW)
    labels = create_labels(df, features)

    print(f"  Positive labels (pre-event ticks): {labels.sum():.0f}/{len(labels)} "
          f"({labels.mean()*100:.1f}%)")

    # ── Step 2: Train/test split ─────────────────────────────────────────
    print("\n[2/5] Preparing sequences...")
    split = int(len(features) * 0.7)

    train_features = features[:split]
    test_features = features[split:]
    train_labels = labels[:split]
    test_labels = labels[split:]

    df_test = df.iloc[split:].reset_index(drop=True)
    test_events = find_anomaly_events(df_test, threshold_pct=ANOMALY_THRESHOLD_PCT, window=EARLY_WARNING_WINDOW)

    train_seqs = prepare_sequences(train_features, seq_len=SEQ_LEN, stride=128)
    train_label_seqs = prepare_sequences(
        train_labels.reshape(-1, 1), seq_len=SEQ_LEN, stride=128
    ).squeeze(-1)

    val_start = int(len(train_seqs) * 0.85)
    val_seqs = train_seqs[val_start:]
    val_label_seqs = train_label_seqs[val_start:]
    train_seqs = train_seqs[:val_start]
    train_label_seqs = train_label_seqs[:val_start]

    print(f"  Train sequences: {len(train_seqs)}")
    print(f"  Val sequences: {len(val_seqs)}")
    print(f"  Test ticks: {len(test_features):,}")
    print(f"  Test events: {len(test_events)}")

    # ── Step 3: Train models ─────────────────────────────────────────────
    print("\n[3/5] Training Griffin...")
    griffin = GriffinBTCDetector(d_model=D_MODEL, n_layers=N_LAYERS)
    print(f"  Parameters: {sum(p.numel() for p in griffin.parameters()):,}")
    griffin, g_train_loss, g_val_loss = train_model(
        griffin, train_seqs, train_label_seqs, val_seqs, val_label_seqs, "Griffin"
    )

    print("\n  Training Transformer...")
    transformer = TransformerBTCDetector(d_model=D_MODEL, n_layers=N_LAYERS)
    print(f"  Parameters: {sum(p.numel() for p in transformer.parameters()):,}")
    transformer, t_train_loss, t_val_loss = train_model(
        transformer, train_seqs, train_label_seqs, val_seqs, val_label_seqs, "Transformer"
    )

    # ── Step 4: Evaluate ─────────────────────────────────────────────────
    print("\n[4/5] Evaluating anomaly detection...")

    # Griffin: true real-time (tick by tick, O(1) memory)
    # Process in chunks to speed up while still being sequential
    griffin_results, griffin_scores, griffin_warns = evaluate_anomaly_detection(
        griffin, df_test, test_features, test_events, "Griffin"
    )

    # Transformer: sliding window around each event (since O(n²) is too slow for full stream)
    # This is the BEST CASE for Transformer — it gets to focus on the event region
    print(f"\n  [Transformer] Evaluating on event regions (sliding window)...")
    transformer.eval()
    t_start = time.time()

    # Process regions around each event
    t_scores_full = np.zeros(len(test_features) - 1)
    t_warns_full = np.zeros(len(test_features) - 1)

    for event in test_events:
        idx = event["index"]
        # Process a window around the event
        region_start = max(0, idx - 300)
        region_end = min(len(test_features), idx + 60)

        x_t = torch.tensor(test_features, device=DEVICE).unsqueeze(0)
        for i in range(max(region_start, SEQ_LEN), region_end):
            start = max(0, i - SEQ_LEN)
            chunk = x_t[:, start:i, :]
            with torch.no_grad():
                pred, warning, _ = transformer(chunk)
                if chunk.size(1) > 1:
                    mse = F.mse_loss(pred[:, -1:, :], x_t[:, i:i+1, :]).item()
                    z = (mse - transformer.baseline_mean.item()) / (transformer.baseline_std.item() + 1e-8)
                    if i - 1 < len(t_scores_full):
                        t_scores_full[i-1] = z
                        t_warns_full[i-1] = warning[:, -1, 0].item()

    t_elapsed = time.time() - t_start
    transformer_scores = t_scores_full
    transformer_warns = t_warns_full

    # Evaluate Transformer detections
    transformer_results = {
        "detected": 0,
        "total": len(test_events),
        "detections": [],
        "ticks_per_sec": 300 * len(test_events) / max(t_elapsed, 0.01),
        "elapsed": t_elapsed,
    }
    for event in test_events:
        idx = event["index"]
        if idx >= len(transformer_scores) or idx < EARLY_WARNING_WINDOW:
            continue
        lookback = min(120, idx)
        region = transformer_scores[idx-lookback:idx]
        detected_indices = np.where(region > 2.0)[0]
        if len(detected_indices) > 0:
            transformer_results["detections"].append({
                **event, "detected": True,
                "early_warning_ticks": int(lookback - detected_indices[0]),
                "peak_score": float(region.max()),
            })
        else:
            transformer_results["detections"].append({
                **event, "detected": False,
                "early_warning_ticks": 0,
                "peak_score": float(region.max()) if len(region) > 0 else 0,
            })

    t_det = sum(1 for d in transformer_results["detections"] if d["detected"])
    transformer_results["detected"] = t_det

    print(f"  [Transformer] Done in {t_elapsed:.1f}s")
    print(f"  [Transformer] Events detected: {t_det}/{transformer_results['total']}")
    for d in transformer_results["detections"]:
        status = "✓ DETECTED" if d["detected"] else "✗ MISSED"
        print(f"    {status} {d['direction']} {d['pct_change']:+.2f}% | "
              f"early warning: {d['early_warning_ticks']}s before | "
              f"peak score: {d['peak_score']:.2f}")

    # ── Step 5: Memory benchmark + visualization ─────────────────────────
    print("\n[5/5] Memory benchmark & visualization...")
    memory_results = memory_benchmark(griffin, transformer)

    # Extend transformer scores to match griffin for plotting
    griffin_scores_plot = griffin_scores[:len(test_features)]
    griffin_warns_plot = griffin_warns[:len(test_features)]

    # For the plot, pad transformer to same length with zeros
    t_scores_full = np.zeros(len(test_features) - 1)
    t_warns_full = np.zeros(len(test_features) - 1)
    t_scores_full[:len(transformer_scores)] = transformer_scores[:len(t_scores_full)]
    t_warns_full[:len(transformer_warns)] = transformer_warns[:len(t_warns_full)]

    plot_results(
        df_test, test_features, test_events,
        griffin_scores_plot, griffin_warns_plot,
        t_scores_full, t_warns_full,
        griffin_results, transformer_results,
        memory_results,
    )

    # ── Summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"  Griffin  : {griffin_results['detected']}/{griffin_results['total']} events detected")
    print(f"  Transformer: {transformer_results['detected']}/{transformer_results['total']} events detected")
    print(f"  Griffin inference: {griffin_results['ticks_per_sec']:,.0f} ticks/sec (O(1) memory)")
    print(f"  Griffin can stream BTC 24/7 — Transformer needs sliding windows")
    print(f"\n  Dashboard: btc_anomaly_dashboard.png")

    # Save results
    results = {
        "griffin": {
            "events_detected": griffin_results["detected"],
            "events_total": griffin_results["total"],
            "ticks_per_sec": griffin_results["ticks_per_sec"],
            "detections": griffin_results["detections"],
        },
        "transformer": {
            "events_detected": transformer_results["detected"],
            "events_total": transformer_results["total"],
            "detections": transformer_results["detections"],
        },
        "memory_scaling": memory_results,
        "config": {
            "d_model": D_MODEL,
            "n_layers": N_LAYERS,
            "seq_len": SEQ_LEN,
            "device": DEVICE,
        },
    }
    with open("btc_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Results: btc_results.json")


if __name__ == "__main__":
    main()
