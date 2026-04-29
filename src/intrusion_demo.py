"""
Griffin Early Warning on NSL-KDD Network Intrusion Dataset.

Real data: 148,517 network connections, labeled normal/attack.
Key insight: attacks leave PRECURSOR PATTERNS in preceding normal traffic.
Griffin learns the rhythm of normal traffic, then predicts:
  "An attack is coming in the next N connections"

Attack types in data:
  DoS:   neptune (SYN flood), smurf, pod, teardrop
  Probe: portsweep, satan, ipsweep, nmap
  R2L:   warezclient, imap, guess_passwd
  U2R:   buffer_overflow, rootkit
"""

import os, math, random, time
import numpy as np
import torch
import torch.nn.functional as F

from griffin_model import GriffinAnomalyDetector
from future_predictor import GriffinPredictor, train_predictor

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
DATA_DIR = os.path.join(os.path.dirname(__file__), "datasets/nsl_kdd")
HORIZON  = 10   # predict attack in next 10 connections

# Attack categories
DOS_ATTACKS   = {"neptune","smurf","pod","teardrop","land","back","apache2","udpstorm"}
PROBE_ATTACKS = {"portsweep","satan","ipsweep","nmap","mscan","saint"}
R2L_ATTACKS   = {"warezclient","imap","ftp_write","guess_passwd","warezmaster","phf","spy","multihop"}
U2R_ATTACKS   = {"buffer_overflow","rootkit","loadmodule","perl","sqlattack","xterm","ps"}
ALL_ATTACKS   = DOS_ATTACKS | PROBE_ATTACKS | R2L_ATTACKS | U2R_ATTACKS

PROTO_MAP   = {"tcp": 0, "udp": 1, "icmp": 2}
FLAG_MAP    = {"SF":0,"S0":1,"REJ":2,"RSTO":3,"RSTR":4,"SH":5,"OTH":6,"S1":7,"S2":8,"S3":9,"S4":10}


def encode_row(row: list) -> list[float]:
    """
    NSL-KDD row → 7-dim normalised feature vector (same shape as ECG).
    Uses the most discriminative features for attack detection.
    """
    proto        = PROTO_MAP.get(row[1], 0) / 2.0
    src_bytes    = math.log1p(float(row[4])) / 20.0
    dst_bytes    = math.log1p(float(row[5])) / 20.0
    duration     = math.log1p(float(row[0])) / 10.0
    serror_rate  = float(row[24])                    # SYN error rate 0-1
    dst_serror   = float(row[37])                    # dst host SYN error rate
    count        = float(row[22]) / 511.0            # connections to same host

    return [proto, src_bytes, dst_bytes, duration, serror_rate, dst_serror, count]


def load_nslkdd(path: str, max_rows: int = 50000):
    """
    Load NSL-KDD CSV → list of (features, label, attack_type) tuples.
    """
    records = []
    with open(path) as f:
        for i, line in enumerate(f):
            if i >= max_rows:
                break
            parts = line.strip().split(",")
            if len(parts) < 42:
                continue
            label      = parts[41].strip()         # "normal", "neptune", etc.
            is_attack  = label in ALL_ATTACKS
            attack_cat = (
                "DoS"   if label in DOS_ATTACKS else
                "Probe" if label in PROBE_ATTACKS else
                "R2L"   if label in R2L_ATTACKS else
                "U2R"   if label in U2R_ATTACKS else
                "normal"
            )
            try:
                feat = encode_row(parts)
                records.append((feat, is_attack, attack_cat, label))
            except (ValueError, IndexError):
                continue
    return records


def make_windows(records, window=30, stride=5, horizon=HORIZON):
    """
    Sliding windows of connections.
    Label for each window position t:
      1 if any of next `horizon` connections is an attack.
    """
    sequences = []
    feats  = [r[0] for r in records]
    labels = [r[1] for r in records]

    for start in range(0, len(feats) - window - horizon, stride):
        f_win = feats[start : start + window]
        # Future label: is there an attack in the horizon after each position?
        l_win = []
        for t in range(window):
            abs_t = start + t
            future = labels[abs_t+1 : abs_t+1+horizon]
            l_win.append(1.0 if any(future) else 0.0)
        sequences.append((f_win, l_win))

    return sequences


def run_intrusion_demo(model, records, start=1000, n_show=80):
    """
    Stream connections one by one, show P(attack in next 10) rising.
    Find a window that contains the transition normal→attack.
    """
    # Find a good window: normal traffic then attack starts
    attack_indices = [i for i, r in enumerate(records) if r[1]]
    if not attack_indices:
        print("No attacks in records!")
        return

    # Find first attack cluster with some normal traffic before it
    for attack_start in attack_indices:
        if attack_start > 30 and not records[attack_start - 15][1]:
            start = max(0, attack_start - 25)
            break

    window = records[start : start + n_show]
    feats  = [r[0] for r in window]
    labels = [r[1] for r in window]
    cats   = [r[2] for r in window]
    types  = [r[3] for r in window]

    # Build future labels
    future_labels = []
    for t in range(len(window)):
        future = labels[t+1 : t+1+HORIZON]
        future_labels.append(1 if any(future) else 0)

    x = torch.tensor(feats, dtype=torch.float32).to(DEVICE)
    model.eval()
    hidden = None

    print(f"\n{'═'*72}")
    print(f"  🌐  GRIFFIN NETWORK INTRUSION PREDICTOR")
    print(f"      P(attack in next {HORIZON} connections) — real NSL-KDD data")
    print(f"{'═'*72}")
    print()
    header = f"  {'#':>4}  {'type':^14}  {'SYNerr':>7}  {'P(atk)':>8}  {'bar':^26}  result"
    print(header)
    print("  " + "─"*len(header))

    first_alert = None
    first_real_attack = next((t for t, r in enumerate(window) if r[1]), None)

    with torch.no_grad():
        for t in range(len(window) - 1):
            x_t    = x[t].unsqueeze(0).unsqueeze(0)
            logit, hidden = model(x_t, hidden)
            prob = torch.sigmoid(logit[0, 0]).item()

            is_attack   = labels[t]
            future_atk  = future_labels[t]
            syn_err     = feats[t][4]

            type_icon = {
                "normal": "🟢 normal      ",
                "DoS":    "🔴 DoS         ",
                "Probe":  "🟡 Probe       ",
                "R2L":    "🟠 R2L         ",
                "U2R":    "🔴 U2R         ",
            }.get(cats[t], "⚪ unknown     ")

            bar_fill = int(min(prob, 1.0) * 26)
            bar_str  = "█"*bar_fill + "░"*(26-bar_fill)

            if prob > 0.70:
                alert = f"🚨 HIGH RISK {prob:.0%}"
                if first_alert is None:
                    first_alert = t
            elif prob > 0.40:
                alert = f"⚠️  moderate  {prob:.0%}"
            else:
                alert = f"   safe      {prob:.0%}"

            mark = " ←── ATTACK STARTS" if t == first_real_attack else ""
            print(f"  {start+t:>4}  {type_icon}  {syn_err:>6.2f}  {prob:>7.1%}  "
                  f"{bar_str}  {alert}{mark}")
            time.sleep(0.03)

    print(f"\n{'═'*72}")
    if first_alert is not None and first_real_attack is not None:
        lead = first_real_attack - first_alert
        if lead > 0:
            print(f"  ✅  Griffin raised HIGH RISK at connection #{start+first_alert}")
            print(f"      First real attack at  connection #{start+first_real_attack}")
            print(f"      → {lead} connections early warning!")
        else:
            print(f"  ⚡  Flagged at the attack boundary (0 lead)")
    print(f"{'═'*72}\n")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    train_path = os.path.join(DATA_DIR, "KDDTrain+.txt")
    test_path  = os.path.join(DATA_DIR, "KDDTest+.txt")

    print("\n  Loading NSL-KDD training data...")
    train_records = load_nslkdd(train_path, max_rows=100000)
    test_records  = load_nslkdd(test_path,  max_rows=22544)

    n_normal = sum(1 for r in train_records if not r[1])
    n_attack = sum(1 for r in train_records if r[1])
    print(f"  Train: {n_normal:,} normal  +  {n_attack:,} attacks")

    # Only train on NORMAL traffic windows — so model learns normal baseline
    normal_records = [r for r in train_records if not r[1]]
    print(f"  Building normal-only windows...")
    train_seqs = make_windows(normal_records, window=30, stride=3)
    print(f"  Windows: {len(train_seqs):,}")

    model = GriffinPredictor(d_model=64, n_layers=2).to(DEVICE)
    train_predictor(model, train_seqs, epochs=25, device=DEVICE)

    # Test: run on the FULL test set (normal + attacks interleaved)
    print(f"\n  Running on test data ({len(test_records):,} connections)...")
    run_intrusion_demo(model, test_records, n_show=80)
