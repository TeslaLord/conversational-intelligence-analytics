"""CSV loader + cleaning + PII hashing.

Notes from dataset exploration:
- Text is real seed sentence + gibberish padding. We keep the first 1-2
  meaningful sentences via a heuristic.
- Timestamps within a `conv_id` are non-monotonic; we trust `turn_index`
  for ordering and flag `is_time_clean` at conversation level.
"""
from __future__ import annotations

import csv
import hashlib
import re

import pandas as pd

REQUIRED_COLUMNS = [
    "conv_id", "turn_index", "role", "text", "timestamp",
    "industry", "product", "issue_type", "language", "channel",
    "customer_name", "agent_name",
    "overall_sentiment", "overall_urgency", "outcome", "primary_intent",
]

# Bump this whenever the cleaning heuristics change so the pipeline knows to
# re-clean previously ingested conversations.
CLEANING_VERSION = "2026-05-23-gibberish-tail-v3"

# Token that looks like gibberish: 5+ chars, all lowercase letters, no vowels-adjacent
# This is intentionally simple; the goal is to truncate the padding tail.
_GIBBERISH = re.compile(r"\b[a-z]{5,}\b")
_SENT_SPLIT = re.compile(r"(?<=[\.!?])\s+")


# A token is "gibberish-ish" if it is a lowercase pure-letter run with very
# few vowels. We deliberately allow short tokens (>=3 chars) because the
# padding includes plenty of `qfx`, `bac`, `scyn`, etc.
def _is_gibberish_token(tok: str) -> bool:
    if len(tok) < 3 or not tok.isalpha() or not tok.islower():
        return False
    vowels = sum(c in "aeiou" for c in tok)
    # Pure consonant runs (qfx, scyn) -> always gibberish.
    if vowels == 0:
        return True
    # Otherwise: low vowel ratio AND no common English bigram.
    ratio = vowels / len(tok)
    return ratio < 0.34


# Common English/Hinglish digraphs that almost never appear in random
# consonant soup. If a token contains any of these AND has at least one
# vowel, treat it as a real word even if its vowel ratio is low.
_REAL_BIGRAMS = (
    "th", "ch", "sh", "wh", "ph", "gh", "ng", "qu", "ck",
    "ee", "oo", "ai", "ou", "ea", "ie", "io", "ion",
    "ing", "ed", "er", "ly", "tion",
)


def _looks_real(tok: str) -> bool:
    low = tok.lower()
    return any(bg in low for bg in _REAL_BIGRAMS)


def clean_text(raw: str) -> str:
    """Strip the gibberish padding tail.

    Algorithm: tokenize on whitespace, then find the first index at which
    we see a RUN of >= 3 consecutive gibberish-ish tokens (no real bigram).
    Everything from that index onward is padding -> drop it. Real prose
    almost never produces three mostly-consonant lowercase tokens in a row,
    while the synthetic padding always does (often 30+ in a row).
    """
    if not isinstance(raw, str) or not raw.strip():
        return ""
    text = raw.strip()
    tokens = text.split()
    if not tokens:
        return ""

    RUN = 3
    cut_at: int | None = None
    streak = 0
    streak_start = 0
    for i, tok in enumerate(tokens):
        bare = tok.strip(",.!?;:\"'()[]{}")
        if _is_gibberish_token(bare) and not _looks_real(bare):
            if streak == 0:
                streak_start = i
            streak += 1
            if streak >= RUN:
                cut_at = streak_start
                break
        else:
            streak = 0

    if cut_at is None:
        cleaned = text
    else:
        cleaned = " ".join(tokens[:cut_at]).strip()
        # Trim any dangling sentence fragment with no terminal punctuation
        # by walking back to the last `.`, `!`, or `?`.
        m = re.search(r"[\.!?](?!.*[\.!?])", cleaned, re.DOTALL)
        if m:
            cleaned = cleaned[: m.end()].strip()

    # Cap at ~400 chars on a sentence boundary if possible.
    if len(cleaned) > 400:
        truncated = cleaned[:400]
        last = max(truncated.rfind("."), truncated.rfind("!"), truncated.rfind("?"))
        if last >= 60:
            truncated = truncated[: last + 1]
        cleaned = truncated.strip()
    return cleaned


def hash_pii(value: str, salt: str) -> str:
    if not isinstance(value, str) or not value:
        return ""
    return hashlib.sha256((salt + "::" + value).encode("utf-8")).hexdigest()[:16]


def load_raw_csv(csv_path) -> pd.DataFrame:
    """Robust read; matches the notebook's working strategy."""
    try:
        df = pd.read_csv(
            csv_path,
            engine="python",
            encoding="utf-8",
            quoting=csv.QUOTE_MINIMAL,
            on_bad_lines="skip",
        )
    except Exception:
        df = pd.read_csv(
            csv_path,
            engine="python",
            encoding="latin1",
            quoting=csv.QUOTE_NONE,
            on_bad_lines="skip",
        )
    df.columns = (
        df.columns.str.strip().str.lower().str.replace("[^a-z0-9_]+", "_", regex=True)
    )
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")
    return df


def prepare_turns(df: pd.DataFrame, salt: str) -> pd.DataFrame:
    """Clean text, hash PII, coerce types. Returns turn-level DataFrame."""
    out = df[REQUIRED_COLUMNS].copy()
    out["turn_index"] = pd.to_numeric(out["turn_index"], errors="coerce").astype("Int64")
    out = out.dropna(subset=["conv_id", "turn_index", "text"]).copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], errors="coerce", utc=True)
    out["clean_text"] = out["text"].astype(str).map(clean_text)
    out = out[out["clean_text"].str.len() > 0].copy()
    out["customer_hash"] = out["customer_name"].astype(str).map(lambda v: hash_pii(v, salt))
    out["agent_hash"] = out["agent_name"].astype(str).map(lambda v: hash_pii(v, salt))
    out = out.drop(columns=["customer_name", "agent_name", "text"])
    out = out.sort_values(["conv_id", "turn_index"]).reset_index(drop=True)
    return out
