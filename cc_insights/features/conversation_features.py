"""Per-conversation aggregates derived from per-turn features."""
from __future__ import annotations

import numpy as np
import pandas as pd


def _slope(y: list[float]) -> float:
    if len(y) < 2:
        return 0.0
    x = np.arange(len(y), dtype="float64")
    yv = np.asarray(y, dtype="float64")
    # least squares slope
    xm = x.mean()
    ym = yv.mean()
    denom = ((x - xm) ** 2).sum()
    if denom == 0:
        return 0.0
    return float(((x - xm) * (yv - ym)).sum() / denom)


def build_conversations(turns: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for conv_id, g in turns.groupby("conv_id", sort=False):
        g = g.sort_values("turn_index")
        traj = g["turn_sentiment_signed"].astype(float).tolist()
        ts = pd.to_datetime(g["timestamp"], utc=True, errors="coerce").dropna()
        is_time_clean = bool(len(ts) >= 2 and ts.is_monotonic_increasing)
        if len(ts) >= 2:
            duration_min = float((ts.max() - ts.min()).total_seconds() / 60.0)
        else:
            duration_min = None
        cust = g[g["role"] == "customer"]
        agent = g[g["role"] == "agent"]
        neg_ratio = float((g["turn_sentiment"] == "negative").mean()) if len(g) else 0.0
        rephrase_count = int(g["is_rephrase"].sum())
        clarifying_count = int(g["is_clarifying_q"].sum())
        outcome = g["outcome"].iloc[0]
        task_completed = int(str(outcome).lower() == "resolved")
        # dissatisfaction in [0,1]
        dissat = float(np.clip(
            0.5 * neg_ratio
            + 0.2 * min(rephrase_count / 3.0, 1.0)
            + 0.3 * (0.0 if task_completed else 1.0),
            0.0, 1.0,
        ))
        latencies = g["response_latency_min"].dropna()
        rows.append({
            "conv_id": conv_id,
            "industry": g["industry"].iloc[0],
            "product": g["product"].iloc[0],
            "issue_type": g["issue_type"].iloc[0],
            "language": g["language"].iloc[0],
            "channel": g["channel"].iloc[0],
            "agent_hash": g["agent_hash"].iloc[0],
            "customer_hash": g["customer_hash"].iloc[0],
            "primary_intent": g["primary_intent"].iloc[0],
            "overall_sentiment": g["overall_sentiment"].iloc[0],
            "overall_urgency": g["overall_urgency"].iloc[0],
            "outcome": outcome,
            "turn_count": int(len(g)),
            "customer_turn_count": int(len(cust)),
            "agent_turn_count": int(len(agent)),
            "duration_minutes": duration_min,
            "is_time_clean": is_time_clean,
            "started_at": ts.min().to_pydatetime() if len(ts) else None,
            "ended_at": ts.max().to_pydatetime() if len(ts) else None,
            "month": ts.min().strftime("%Y-%m") if len(ts) else None,
            "sentiment_start": traj[0] if traj else 0.0,
            "sentiment_end": traj[-1] if traj else 0.0,
            "sentiment_min": float(min(traj)) if traj else 0.0,
            "sentiment_slope": _slope(traj),
            "sentiment_volatility": float(np.std(traj)) if traj else 0.0,
            "negative_turn_ratio": neg_ratio,
            "rephrase_count": rephrase_count,
            "clarifying_question_count": clarifying_count,
            "task_completed": task_completed,
            "dissatisfaction_score": dissat,
            "avg_response_latency_min": float(latencies.mean()) if len(latencies) else None,
            "p95_response_latency_min": float(np.percentile(latencies, 95)) if len(latencies) else None,
            # placeholders filled by llm_tagger
            "summary": None,
            "topic_tags": None,
            "agent_empathy_mean": None,
        })
    return pd.DataFrame(rows)
