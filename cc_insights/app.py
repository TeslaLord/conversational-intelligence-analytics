"""Wire-up factory used by API, UI, and eval."""
from __future__ import annotations

from .agent.orchestrator import Orchestrator
from .agent.planner import Planner
from .agent.synthesizer import Synthesizer
from .agent.tools.conv_detail_tool import ConvDetailTool
from .agent.tools.retrieval_tool import RetrievalTool
from .agent.tools.sql_tool import SQLTool
from .config import get_settings
from .models.embeddings import Embedder
from .models.llm import LLMClient
from .storage.duckdb_store import Warehouse
from .storage.session_store import SessionStore
from .storage.vector_store import VectorStore


_ENUM_COLUMNS = (
    "industry", "product", "primary_intent", "outcome",
    "language", "channel", "overall_sentiment",
)


def _load_enum_values(warehouse: Warehouse) -> dict[str, list[str]]:
    """Pull distinct values for low-cardinality columns so the planner
    cannot hallucinate filter values (e.g. '5G' instead of '5G Upgrade')."""
    out: dict[str, list[str]] = {}
    for col in _ENUM_COLUMNS:
        try:
            df = warehouse.query(
                f"SELECT DISTINCT {col} AS v FROM conversations "
                f"WHERE {col} IS NOT NULL ORDER BY {col}"
            )
            out[col] = [str(v) for v in df["v"].tolist()]
        except Exception:
            # Column missing or warehouse not yet populated -> skip.
            continue
    return out


def build_orchestrator():
    s = get_settings()
    if not s.duckdb_path.exists():
        raise RuntimeError(
            f"DuckDB warehouse missing at {s.duckdb_path}. Run `python -m cc_insights.pipeline` first."
        )
    llm = LLMClient(
        api_key=s.LLM_API_KEY,
        base_url=s.LLM_BASE_URL,
        cache_path=s.llm_cache_path,
    )
    embedder = Embedder(s.EMBEDDING_MODEL)
    warehouse = Warehouse(s.duckdb_path)
    vector = VectorStore(s.chroma_dir)
    enum_values = _load_enum_values(warehouse)

    return Orchestrator(
        planner=Planner(llm, s.SYNTH_LLM_MODEL, valid_values=enum_values),
        synthesizer=Synthesizer(llm, s.SYNTH_LLM_MODEL),
        sql=SQLTool(warehouse),
        retrieval=RetrievalTool(vector, embedder),
        conv_detail=ConvDetailTool(warehouse),
        traces_dir=s.traces_dir,
        top_k_final=s.TOP_K_FINAL,
    )


def build_session_store() -> SessionStore:
    s = get_settings()
    return SessionStore(s.session_db_path)
