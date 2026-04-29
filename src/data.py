"""
Data extraction from agentwatch TiDB database.
Falls back to synthetic data if DB is unavailable.

The key sequence per session:
    spans ordered by startTime → each span = one time step
    Features: [span_type, duration_ms, tokens_used, cost_cents,
               risk_level, status, step_index]
"""

import math
import random
import numpy as np
from dataclasses import dataclass

# ── DB Config (fill in DB_PASSWORD to use real data) ─────────────────────────
DB_CONFIG = {
    "host": "gateway04.us-east-1.prod.aws.tidbcloud.com",
    "port": 4000,
    "user": "3SwCYMQVVRzbDKd.e85b37e81e70",
    "password": "",   # ← paste TiDB password here
    "database": "JY6b5ecv6BBEKRuuY592pX",
    "ssl": {"ssl_verify_cert": False},
}

# ── Feature encoding maps ─────────────────────────────────────────────────────
SPAN_TYPES = {"llm": 0, "tool": 1, "decision": 2, "action": 3, "system": 4}
RISK_LEVELS = {"safe": 0, "warning": 1, "danger": 2}
STATUS_MAP  = {"completed": 0, "error": 1, "running": 0}


@dataclass
class SpanFeatures:
    """Normalised feature vector for a single span."""
    span_type:   float   # 0-4 / 4  (normalised to 0-1)
    duration:    float   # log1p(ms) / 10
    tokens:      float   # log1p(tokens) / 10
    cost:        float   # log1p(cents) / 5
    risk_level:  float   # 0, 0.5, 1.0
    status:      float   # 0 = ok, 1 = error
    step_index:  float   # position / max_steps

    def to_vector(self) -> list[float]:
        return [
            self.span_type,
            self.duration,
            self.tokens,
            self.cost,
            self.risk_level,
            self.status,
            self.step_index,
        ]


def _encode_span(row: dict, step_idx: int, max_steps: int) -> SpanFeatures:
    """Convert a raw DB row → normalised SpanFeatures."""
    return SpanFeatures(
        span_type  = SPAN_TYPES.get(row.get("type", "llm"), 0) / 4.0,
        duration   = math.log1p(row.get("durationMs", 0)) / 10.0,
        tokens     = math.log1p(row.get("tokensUsed", 0)) / 10.0,
        cost       = math.log1p(row.get("costCents", 0)) / 5.0,
        risk_level = RISK_LEVELS.get(row.get("riskLevel", "safe"), 0) / 2.0,
        status     = STATUS_MAP.get(row.get("status", "completed"), 0),
        step_index = step_idx / max(max_steps, 1),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Real DB extraction
# ─────────────────────────────────────────────────────────────────────────────

def load_from_db(min_spans: int = 3, limit_sessions: int = 2000) -> tuple[list, list]:
    """
    Pull sessions + spans from TiDB.
    Returns (normal_sessions, anomalous_sessions) where each session
    is a list of feature vectors.
    """
    import pymysql

    conn = pymysql.connect(
        host=DB_CONFIG["host"],
        port=DB_CONFIG["port"],
        user=DB_CONFIG["user"],
        password=DB_CONFIG["password"],
        database=DB_CONFIG["database"],
        ssl=DB_CONFIG["ssl"],
        connect_timeout=10,
    )

    with conn.cursor(pymysql.cursors.DictCursor) as cur:
        # Pull sessions with their risk level
        cur.execute(f"""
            SELECT sessionId, riskLevel, success, createdAt
            FROM agent_sessions
            WHERE status = 'completed'
            ORDER BY createdAt ASC
            LIMIT {limit_sessions}
        """)
        sessions = cur.fetchall()

        normal, anomalous = [], []

        for sess in sessions:
            sid = sess["sessionId"]

            cur.execute("""
                SELECT type, durationMs, tokensUsed, costCents,
                       riskLevel, status
                FROM spans
                WHERE sessionId = %s
                ORDER BY startTime ASC
            """, (sid,))
            span_rows = cur.fetchall()

            if len(span_rows) < min_spans:
                continue

            features = [
                _encode_span(r, i, len(span_rows)).to_vector()
                for i, r in enumerate(span_rows)
            ]

            if sess["riskLevel"] in ("warning", "danger") or not sess.get("success", True):
                anomalous.append(features)
            else:
                normal.append(features)

    conn.close()
    print(f"[DB] Loaded {len(normal)} normal + {len(anomalous)} anomalous sessions")
    return normal, anomalous


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic data (mirrors real schema perfectly)
# ─────────────────────────────────────────────────────────────────────────────

def _normal_session(n_spans: int = None) -> list[list[float]]:
    """
    Generate a realistic 'healthy' agent session.
    Pattern: llm → tool → tool → decision → action → llm → ... → completed
    Stable durations, low risk, no errors.
    """
    if n_spans is None:
        n_spans = random.randint(4, 15)

    # Typical healthy sequence pattern
    type_sequence = [0] + [random.choice([0, 1, 2]) for _ in range(n_spans - 2)] + [0]

    spans = []
    for i, t in enumerate(type_sequence):
        risk = random.choices([0, 1, 2], weights=[0.92, 0.07, 0.01])[0]
        err  = random.random() < 0.03

        # Normal distributions for timing (low variance)
        base_duration = {0: 1200, 1: 800, 2: 150, 3: 500, 4: 50}[t]
        duration = max(10, random.gauss(base_duration, base_duration * 0.15))

        base_tokens = {0: 1200, 1: 50, 2: 200, 3: 100, 4: 10}[t]
        tokens = max(0, random.gauss(base_tokens, base_tokens * 0.10))

        f = SpanFeatures(
            span_type  = t / 4.0,
            duration   = math.log1p(duration) / 10.0,
            tokens     = math.log1p(tokens) / 10.0,
            cost       = math.log1p(tokens * 0.002) / 5.0,
            risk_level = risk / 2.0,
            status     = float(err),
            step_index = i / n_spans,
        )
        spans.append(f.to_vector())
    return spans


def _anomalous_session(anomaly_at: int = None) -> tuple[list[list[float]], int]:
    """
    Generate a session that 'goes wrong' at a specific point.

    Types of anomalies (like what we see in agentwatch):
    - Sudden spike in duration (agent gets stuck)
    - Risk level jumps to warning/danger
    - Token explosion (runaway loop)
    - Error cascade (status=error propagates)

    Returns: (features, anomaly_start_index)
    """
    n_spans = random.randint(8, 20)
    if anomaly_at is None:
        # Anomaly happens in the first half — so Griffin can catch it early
        anomaly_at = random.randint(3, n_spans // 2)

    anomaly_type = random.choice(["stuck", "risk_spike", "token_explosion", "error_cascade"])

    spans = []
    for i in range(n_spans):
        t = random.choice([0, 1, 2, 3, 4])
        is_anomaly = i >= anomaly_at

        if not is_anomaly:
            # Normal phase — same as healthy session
            risk = random.choices([0, 1, 2], weights=[0.92, 0.07, 0.01])[0]
            err  = random.random() < 0.03
            base_dur = {0: 1200, 1: 800, 2: 150, 3: 500, 4: 50}[t]
            duration = max(10, random.gauss(base_dur, base_dur * 0.15))
            base_tok = {0: 1200, 1: 50, 2: 200, 3: 100, 4: 10}[t]
            tokens = max(0, random.gauss(base_tok, base_tok * 0.10))
        else:
            # Anomaly phase
            if anomaly_type == "stuck":
                # Duration 10-20x normal
                base_dur = {0: 1200, 1: 800, 2: 150, 3: 500, 4: 50}[t]
                duration = base_dur * random.uniform(10, 20)
                tokens   = random.gauss(1200, 100)
                risk     = random.choice([1, 2])
                err      = random.random() < 0.4

            elif anomaly_type == "risk_spike":
                base_dur = {0: 1200, 1: 800, 2: 150, 3: 500, 4: 50}[t]
                duration = random.gauss(base_dur, base_dur * 0.2)
                tokens   = random.gauss(1200, 200)
                risk     = 2  # always danger
                err      = random.random() < 0.3

            elif anomaly_type == "token_explosion":
                # Runaway — like the stepCountIs bug in TESTING_NOTES
                duration = random.gauss(1500, 200)
                tokens   = random.gauss(8000, 1000)   # 6-8x normal
                risk     = random.choice([1, 2])
                err      = False

            else:  # error_cascade
                base_dur = {0: 1200, 1: 800, 2: 150, 3: 500, 4: 50}[t]
                duration = random.gauss(base_dur, base_dur * 0.5)
                tokens   = random.gauss(800, 200)
                risk     = random.choice([1, 2])
                err      = True  # every span errors

        f = SpanFeatures(
            span_type  = t / 4.0,
            duration   = math.log1p(max(0, duration)) / 10.0,
            tokens     = math.log1p(max(0, tokens)) / 10.0,
            cost       = math.log1p(max(0, tokens * 0.002)) / 5.0,
            risk_level = risk / 2.0,
            status     = float(err),
            step_index = i / n_spans,
        )
        spans.append(f.to_vector())

    return spans, anomaly_at


def generate_synthetic_dataset(
    n_normal: int = 800,
    n_anomalous: int = 200,
    seed: int = 42,
) -> tuple[list, list, list]:
    """
    Generate synthetic dataset that mirrors agentwatch spans data.

    Returns:
        normal_sessions:    list of span-feature-lists (all healthy)
        anomalous_sessions: list of span-feature-lists (contain drift)
        anomaly_indices:    where each anomalous session starts drifting
    """
    random.seed(seed)
    np.random.seed(seed)

    print(f"[Synthetic] Generating {n_normal} normal + {n_anomalous} anomalous sessions...")

    normal = [_normal_session() for _ in range(n_normal)]

    anomalous, anomaly_at = [], []
    for _ in range(n_anomalous):
        sess, idx = _anomalous_session()
        anomalous.append(sess)
        anomaly_at.append(idx)

    print(f"[Synthetic] Done. Avg session length: "
          f"{np.mean([len(s) for s in normal + anomalous]):.1f} spans")

    return normal, anomalous, anomaly_at


def load_data(use_real_db: bool = False) -> tuple[list, list, list | None]:
    """
    Load data from DB or fall back to synthetic.

    Returns:
        normal_sessions, anomalous_sessions, anomaly_indices (None for real DB)
    """
    if use_real_db and DB_CONFIG["password"]:
        try:
            normal, anomalous = load_from_db()
            return normal, anomalous, None
        except Exception as e:
            print(f"[DB] Failed ({e}), falling back to synthetic data")

    return generate_synthetic_dataset()
