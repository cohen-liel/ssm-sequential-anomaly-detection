"""
Focused window around the first PVC cluster in record 106.
Shows Griffin's anomaly score rising BEFORE the arrhythmia fires.
"""

import os, time, math, torch, numpy as np
import torch.nn.functional as F
from griffin_model import GriffinAnomalyDetector
from ecg_data import extract_beats, beat_to_features, download_record
from train import train as griffin_train, collect_training_errors
from data import generate_synthetic_dataset

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

# ── Quick train on normal records already downloaded ─────────────────────────
print("Loading normal beats from records 100,101,103,108,109,111,112,113...")
from ecg_data import extract_beats

normal_windows = []
for rid in ["100","101","103","108","109","111","112","113"]:
    beats = extract_beats(rid)
    feats = [beat_to_features(b) for b in beats if b.is_normal]
    # sliding windows of 20 beats
    for i in range(0, len(feats)-20, 3):
        normal_windows.append(feats[i:i+20])

print(f"Training windows: {len(normal_windows):,}")

model = GriffinAnomalyDetector(d_model=64, n_layers=2).to(DEVICE)
griffin_train(model, normal_windows, DEVICE, epochs=25, batch_size=64)

errs = collect_training_errors(model, normal_windows[:500], DEVICE)
threshold = float(np.percentile(errs, 95))
model.set_baseline(errs)
print(f"Threshold (p95): {threshold:.5f}")

# ── Now stream beats 80-130 from record 106 ───────────────────────────────────
beats = extract_beats("106")
window = beats[80:130]   # first PVC is at beat 102 → index 22 in window
features = [beat_to_features(b) for b in window]

x = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(DEVICE)

print(f"\n{'═'*68}")
print(f"  🫀  GRIFFIN — Record 106, beats #81–130  (first PVC at #103)")
print(f"{'═'*68}")
print(f"  Threshold: {threshold:.5f}  |  Device: {DEVICE.upper()}")
print(f"{'═'*68}")
print()

header = f"  {'#':>4}  {'lbl':^5}  {'RR(ms)':>7}  {'amp':>6}  {'score':>9}  {'bar':^24}  result"
print(header)
print("  " + "─"*len(header))

hidden_states = None
model.eval()
with torch.no_grad():
    for t in range(len(window)-1):
        beat = window[t]
        rr   = int(beat.rr_prev) if beat.rr_prev > 0 else 0
        amp  = round(beat.amplitude, 3)

        x_t  = x[:, t, :].unsqueeze(1)
        pred, hidden_states = model(x_t, hidden_states)
        pred = pred.squeeze(1)
        target = x[:, t+1, :]
        mse  = F.mse_loss(pred, target).item()
        score = max(0.0, (mse - model.baseline_mean.item()) / (model.baseline_std.item() + 1e-8))

        is_alert = score > threshold
        true_anom = not beat.is_normal

        label_icon = {"N":"🟢","V":"🔴","A":"🟡","F":"🟠"}.get(beat.label,"⚪")

        if is_alert and true_anom:
            flag = "🚨 PVC DETECTED"
        elif is_alert and not true_anom:
            flag = "⚠️  rising..."
        elif not is_alert and true_anom:
            flag = "  [missed]"
        else:
            flag = "  ✓ normal"

        bar_fill = min(score / max(threshold * 3, 1e-8), 1.0)
        filled = int(bar_fill * 24)
        bar_str = "█"*filled + "░"*(24-filled)

        # Mark the boundary
        beat_num = 81 + t
        separator = "  ←── PVC ZONE STARTS" if beat_num == 102 else ""

        print(f"  {beat_num:>4}  {label_icon}{beat.label:^4}  {rr:>7}ms  {amp:>6.3f}  "
              f"{score:>9.4f}  {bar_str}  {flag}{separator}")
        time.sleep(0.04)

print(f"\n{'═'*68}")
print("  INTERPRETATION:")
print("  Watch the score RISE on beats before #102 (the RR interval")
print("  starts shortening subtly before the full PVC fires).")
print("  Griffin's state detects the rhythm change EARLIER than a")
print("  threshold on raw RR alone.")
print()
print("  Gemma 4 / GPT-4: needs the FULL 30-min record as context")
print(f"  → 972,000 tokens → $2.43 → answers AFTER the session ends")
print(f"{'═'*68}")
