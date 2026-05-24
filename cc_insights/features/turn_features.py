"""Per-turn feature engineering.

Adds:
- turn_sentiment, turn_sentiment_score, turn_sentiment_signed
- is_rephrase  (customer turn very similar to its previous customer turn)
- is_clarifying_q (agent turn that ends with '?')
- response_latency_min (gap to previous opposite-role turn, NaN if not clean)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..models.sentiment import SentimentClassifier


_CLARIFY_HINTS = ("?", "could you", "can you confirm", "can you share", "please confirm")


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def add_turn_features(
    turns: pd.DataFrame,
    sentiment: SentimentClassifier,
    embedder,  # cc_insights.models.embeddings.Embedder
) -> pd.DataFrame:
    df = turns.copy()
    # Sentiment in batch over clean_text
    preds = sentiment.predict_batch(df["clean_text"].tolist())
    df["turn_sentiment"] = [p.label for p in preds]
    df["turn_sentiment_score"] = [p.score for p in preds]
    df["turn_sentiment_signed"] = [p.signed for p in preds]

    # Embeddings for rephrase detection (also reused later if desired)
    print(f"  embedding {len(df)} turns ...")
    emb = embedder.encode(df["clean_text"].tolist(), show_progress=True)

    # Per-conversation features (rephrase, clarifying, latency)
    is_rephrase = np.zeros(len(df), dtype=bool)
    is_clarifying = np.zeros(len(df), dtype=bool)
    latency = np.full(len(df), np.nan, dtype="float64")

    # group indices by conv_id preserving order
    df = df.reset_index(drop=True)
    for _, grp in df.groupby("conv_id", sort=False):
        idxs = grp.index.to_list()
        last_customer_idx: int | None = None
        prev_idx: int | None = None
        prev_role: str | None = None
        for i in idxs:
            role = df.at[i, "role"]
            # clarifying question (agent)
            if role == "agent":
                txt = df.at[i, "clean_text"].lower()
                if any(h in txt for h in _CLARIFY_HINTS):
                    is_clarifying[i] = True
            # rephrase (customer repeating themselves)
            if role == "customer" and last_customer_idx is not None:
                sim = _cosine(emb[last_customer_idx], emb[i])
                if sim > 0.80:
                    is_rephrase[i] = True
            if role == "customer":
                last_customer_idx = i
            # latency between adjacent opposite-role turns
            if prev_idx is not None and prev_role is not None and prev_role != role:
                t_cur = df.at[i, "timestamp"]
                t_prev = df.at[prev_idx, "timestamp"]
                if pd.notna(t_cur) and pd.notna(t_prev):
                    delta = (t_cur - t_prev).total_seconds() / 60.0
                    if delta >= 0:  # only count monotonic gaps
                        latency[i] = delta
            prev_idx = i
            prev_role = role

    df["is_rephrase"] = is_rephrase
    df["is_clarifying_q"] = is_clarifying
    df["response_latency_min"] = latency
    return df, emb
