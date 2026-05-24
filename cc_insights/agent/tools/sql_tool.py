"""Parameterized DuckDB queries -- NO free-form text-to-SQL.

The planner picks one of these intents; the tool builds a safe SQL string
from typed parameters. We expose the SQL in the result for transparency.
"""
from __future__ import annotations

from ...schemas import AggregateResult, AggregateRow
from ...storage.duckdb_store import Warehouse


_ALLOWED_FILTER_COLS = {
    "industry", "product", "primary_intent", "outcome",
    "language", "channel", "overall_sentiment", "overall_urgency",
    "month", "agent_hash",
}


def _build_where(filters: dict) -> tuple[str, list]:
    clauses: list[str] = []
    params: list = []
    for col, val in filters.items():
        if col not in _ALLOWED_FILTER_COLS:
            continue
        if isinstance(val, (list, tuple)):
            placeholders = ",".join("?" for _ in val)
            clauses.append(f"{col} IN ({placeholders})")
            params.extend(val)
        else:
            clauses.append(f"{col} = ?")
            params.append(val)
    return (" WHERE " + " AND ".join(clauses) if clauses else ""), params


class SQLTool:
    def __init__(self, warehouse: Warehouse):
        self._wh = warehouse

    def top_topics_by_sentiment(
        self, sentiment: str, filters: dict, limit: int = 5
    ) -> AggregateResult:
        where_sql, params = _build_where({**filters, "overall_sentiment": sentiment})
        sql = (
            "SELECT industry, product, primary_intent, COUNT(*) AS n_convs, "
            "AVG(dissatisfaction_score) AS avg_dissatisfaction "
            f"FROM conversations{where_sql} "
            "GROUP BY industry, product, primary_intent "
            "ORDER BY n_convs DESC LIMIT ?"
        )
        df = self._wh.query(sql, params + [limit])
        return _df_to_agg(sql, df, group_cols=["industry", "product", "primary_intent"])

    def intent_success(self, filters: dict, limit: int = 20) -> AggregateResult:
        # Filter must be applied to base table, not the materialized one
        where_sql, params = _build_where(filters)
        sql = (
            "SELECT primary_intent, COUNT(*) AS n_convs, "
            "1.0 * SUM(task_completed)/COUNT(*) AS resolution_rate, "
            "AVG(turn_count) AS avg_turns, MIN(turn_count) AS min_turns, MAX(turn_count) AS max_turns, "
            "AVG(dissatisfaction_score) AS avg_dissatisfaction "
            f"FROM conversations{where_sql} "
            "GROUP BY primary_intent ORDER BY n_convs DESC LIMIT ?"
        )
        df = self._wh.query(sql, params + [limit])
        return _df_to_agg(sql, df, group_cols=["primary_intent"])

    def domain_intent_success(self, filters: dict, limit: int = 30) -> AggregateResult:
        where_sql, params = _build_where(filters)
        sql = (
            "SELECT industry, product, primary_intent, COUNT(*) AS n_convs, "
            "1.0 * SUM(task_completed)/COUNT(*) AS resolution_rate, "
            "AVG(dissatisfaction_score) AS avg_dissatisfaction "
            f"FROM conversations{where_sql} "
            "GROUP BY industry, product, primary_intent "
            "ORDER BY n_convs DESC LIMIT ?"
        )
        df = self._wh.query(sql, params + [limit])
        return _df_to_agg(sql, df, group_cols=["industry", "product", "primary_intent"])

    def agents_needing_coaching(self, filters: dict, limit: int = 10) -> AggregateResult:
        where_sql, params = _build_where(filters)
        sql = (
            "SELECT agent_hash, COUNT(*) AS n_convs, "
            "AVG(agent_empathy_mean) AS avg_empathy, "
            "1.0 * SUM(task_completed)/COUNT(*) AS resolution_rate, "
            "AVG(sentiment_end - sentiment_start) AS avg_sentiment_lift, "
            "AVG(dissatisfaction_score) AS avg_dissatisfaction "
            f"FROM conversations{where_sql} "
            "GROUP BY agent_hash HAVING COUNT(*) >= 3 "
            "ORDER BY avg_empathy ASC NULLS LAST, avg_dissatisfaction DESC LIMIT ?"
        )
        df = self._wh.query(sql, params + [limit])
        return _df_to_agg(sql, df, group_cols=["agent_hash"])

    def topic_frequency(self, filters: dict, limit: int = 10) -> AggregateResult:
        where_sql, params = _build_where(filters)
        sql = (
            "SELECT primary_intent, industry, product, COUNT(*) AS n_convs, "
            "AVG(dissatisfaction_score) AS avg_dissatisfaction, "
            "1.0 * SUM(task_completed)/COUNT(*) AS resolution_rate "
            f"FROM conversations{where_sql} "
            "GROUP BY primary_intent, industry, product "
            "ORDER BY avg_dissatisfaction DESC, n_convs DESC LIMIT ?"
        )
        df = self._wh.query(sql, params + [limit])
        return _df_to_agg(sql, df, group_cols=["primary_intent", "industry", "product"])


def _df_to_agg(sql, df, group_cols):
    rows: list[AggregateRow] = []
    for _, r in df.iterrows():
        rec = r.to_dict()
        group = {k: rec.pop(k) for k in group_cols if k in rec}
        metrics = {k: (None if _isna(v) else v) for k, v in rec.items()}
        rows.append(AggregateRow(group=group, metrics=metrics))
    return AggregateResult(sql=sql.strip(), rows=rows, row_count=len(rows))


def _isna(v):
    try:
        import math
        return v is None or (isinstance(v, float) and math.isnan(v))
    except Exception:
        return v is None
