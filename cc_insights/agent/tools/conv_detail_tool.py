"""Single-conversation deep dive."""
from __future__ import annotations

import json

from ...schemas import ConvDetail, TurnView
from ...storage.duckdb_store import Warehouse


class ConvDetailTool:
    def __init__(self, warehouse: Warehouse):
        self._wh = warehouse

    def get(self, conv_id: str) -> ConvDetail:
        conv = self._wh.query(
            "SELECT * FROM conversations WHERE conv_id = ?", [conv_id]
        )
        if conv.empty:
            raise KeyError(f"conv_id not found: {conv_id}")
        c = conv.iloc[0].to_dict()
        turns = self._wh.query(
            "SELECT turn_index, role, clean_text, turn_sentiment, turn_sentiment_score, "
            "turn_sentiment_signed, timestamp FROM turns WHERE conv_id = ? "
            "ORDER BY turn_index",
            [conv_id],
        )
        trajectory = turns["turn_sentiment_signed"].astype(float).tolist()
        tag_raw = c.get("topic_tags")
        try:
            tags = json.loads(tag_raw) if isinstance(tag_raw, str) else (tag_raw or [])
        except Exception:
            tags = []
        return ConvDetail(
            conv_id=conv_id,
            industry=c.get("industry"),
            product=c.get("product"),
            primary_intent=c.get("primary_intent"),
            outcome=c.get("outcome"),
            overall_sentiment=c.get("overall_sentiment"),
            overall_urgency=c.get("overall_urgency"),
            turn_count=int(c.get("turn_count") or 0),
            duration_minutes=c.get("duration_minutes"),
            is_time_clean=bool(c.get("is_time_clean")),
            sentiment_trajectory=trajectory,
            summary=c.get("summary"),
            topic_tags=tags if isinstance(tags, list) else [],
            turns=[
                TurnView(
                    turn_index=int(r["turn_index"]),
                    role=str(r["role"]),
                    text=str(r["clean_text"]),
                    sentiment=r["turn_sentiment"],
                    sentiment_score=float(r["turn_sentiment_score"])
                    if r["turn_sentiment_score"] is not None else None,
                    timestamp=r["timestamp"],
                )
                for _, r in turns.iterrows()
            ],
        )
