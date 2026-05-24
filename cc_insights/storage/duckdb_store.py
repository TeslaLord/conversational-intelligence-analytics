"""DuckDB warehouse: loads turns + conversations + materialized analytic views."""
from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd


_MATERIALIZED_SQL = [
    # intent success ratio
    """
    CREATE OR REPLACE TABLE intent_stats AS
    SELECT primary_intent,
           COUNT(*) AS n_convs,
           SUM(task_completed) AS n_resolved,
           1.0 * SUM(task_completed) / COUNT(*) AS resolution_rate,
           AVG(turn_count) AS avg_turns,
           MIN(turn_count) AS min_turns,
           MAX(turn_count) AS max_turns,
           AVG(dissatisfaction_score) AS avg_dissatisfaction
    FROM conversations
    GROUP BY primary_intent;
    """,
    # domain x intent
    """
    CREATE OR REPLACE TABLE domain_intent_stats AS
    SELECT industry, product, primary_intent,
           COUNT(*) AS n_convs,
           1.0 * SUM(task_completed) / COUNT(*) AS resolution_rate,
           AVG(dissatisfaction_score) AS avg_dissatisfaction
    FROM conversations
    GROUP BY industry, product, primary_intent;
    """,
    # agent scorecard
    """
    CREATE OR REPLACE TABLE agent_scorecard AS
    SELECT agent_hash,
           COUNT(*) AS n_convs,
           1.0 * SUM(task_completed) / COUNT(*) AS resolution_rate,
           AVG(agent_empathy_mean) AS avg_empathy,
           AVG(sentiment_end - sentiment_start) AS avg_sentiment_lift,
           AVG(duration_minutes) FILTER (WHERE is_time_clean) AS avg_handle_time_min,
           AVG(dissatisfaction_score) AS avg_dissatisfaction
    FROM conversations
    WHERE agent_hash IS NOT NULL AND agent_hash <> ''
    GROUP BY agent_hash;
    """,
    # monthly topic / sentiment rollup (for "last month" style questions)
    """
    CREATE OR REPLACE TABLE month_topic_stats AS
    SELECT month, industry, product, primary_intent, overall_sentiment,
           COUNT(*) AS n_convs,
           AVG(dissatisfaction_score) AS avg_dissatisfaction
    FROM conversations
    WHERE month IS NOT NULL
    GROUP BY month, industry, product, primary_intent, overall_sentiment;
    """,
]


class Warehouse:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def connect(self) -> duckdb.DuckDBPyConnection:
        return duckdb.connect(str(self.path))

    def load(self, turns: pd.DataFrame, conversations: pd.DataFrame) -> None:
        """Replace-all load. Used for one-shot/full rebuilds."""
        con = self.connect()
        try:
            con.register("turns_df", turns)
            con.register("conv_df", conversations)
            con.execute("CREATE OR REPLACE TABLE turns AS SELECT * FROM turns_df")
            con.execute("CREATE OR REPLACE TABLE conversations AS SELECT * FROM conv_df")
            self._rebuild_views(con)
            con.execute("CREATE INDEX IF NOT EXISTS idx_turns_conv ON turns(conv_id)")
        finally:
            con.close()

    def upsert(self, turns: pd.DataFrame, conversations: pd.DataFrame) -> None:
        """Append a new batch and rebuild materialized views.

        If base tables don't exist yet, this behaves like `load`. Otherwise it
        deletes any pre-existing rows for the incoming `conv_id`s (idempotent
        re-runs) and inserts the new rows.
        """
        con = self.connect()
        try:
            con.register("turns_df", turns)
            con.register("conv_df", conversations)
            tables = {
                r[0] for r in con.execute(
                    "SELECT table_name FROM information_schema.tables WHERE table_schema='main'"
                ).fetchall()
            }
            if "turns" not in tables or "conversations" not in tables:
                con.execute("CREATE OR REPLACE TABLE turns AS SELECT * FROM turns_df")
                con.execute("CREATE OR REPLACE TABLE conversations AS SELECT * FROM conv_df")
            else:
                con.execute(
                    "DELETE FROM turns WHERE conv_id IN (SELECT conv_id FROM turns_df)"
                )
                con.execute(
                    "DELETE FROM conversations WHERE conv_id IN (SELECT conv_id FROM conv_df)"
                )
                con.execute("INSERT INTO turns SELECT * FROM turns_df")
                con.execute("INSERT INTO conversations SELECT * FROM conv_df")
            self._rebuild_views(con)
            con.execute("CREATE INDEX IF NOT EXISTS idx_turns_conv ON turns(conv_id)")
        finally:
            con.close()

    @staticmethod
    def _rebuild_views(con: duckdb.DuckDBPyConnection) -> None:
        for sql in _MATERIALIZED_SQL:
            con.execute(sql)

    def query(self, sql: str, params: tuple | list | None = None) -> pd.DataFrame:
        con = self.connect()
        try:
            cur = con.execute(sql, params or [])
            return cur.fetchdf()
        finally:
            con.close()
