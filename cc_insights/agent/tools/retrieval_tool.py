"""Vector retrieval over conv_summary (default) with metadata filters."""
from __future__ import annotations

import logging

from ...models.embeddings import Embedder
from ...schemas import RetrievalResult, RetrievedChunk
from ...storage.vector_store import VectorStore


log = logging.getLogger(__name__)


_ALLOWED_FILTER_KEYS = {
    "industry", "product", "primary_intent", "outcome",
    "language", "channel", "overall_sentiment", "month", "agent_hash",
}


class RetrievalTool:
    def __init__(self, vector: VectorStore, embedder: Embedder):
        self._vec = vector
        self._emb = embedder

    @staticmethod
    def _build_where(filters: dict) -> dict | None:
        def _clause(v):
            if isinstance(v, (list, tuple, set)):
                vals = [x for x in v if x is not None]
                if len(vals) == 0:
                    return None
                if len(vals) == 1:
                    return vals[0]
                return {"$in": list(vals)}
            return v

        clauses = {}
        for kk, vv in (filters or {}).items():
            if kk not in _ALLOWED_FILTER_KEYS:
                continue
            c = _clause(vv)
            if c is not None:
                clauses[kk] = c
        if len(clauses) == 0:
            return None
        if len(clauses) == 1:
            (only_k, only_v), = clauses.items()
            return {only_k: only_v}
        return {"$and": [{kk: vv} for kk, vv in clauses.items()]}

    def search(
        self,
        query: str,
        filters: dict,
        k: int,
        collection: str = "conv_summary",
    ) -> RetrievalResult:
        qe = self._emb.encode([query])[0]
        where = self._build_where(filters)

        # Try in order: (1) requested collection with filters, (2) same with
        # no filters, (3) highlights with no filters. This rescues the common
        # failure mode where the planner over-constrained the where-clause or
        # the user asked about something that only appears in raw turn text.
        attempts: list[tuple[str, dict | None]] = []
        attempts.append((collection, where))
        if where is not None:
            attempts.append((collection, None))
        if collection != "conv_highlight":
            attempts.append(("conv_highlight", None))

        hits: list[dict] = []
        used_collection = collection
        used_where = where
        for coll, w in attempts:
            hits = self._vec.query(coll, qe, k=k, where=w)
            if hits:
                used_collection = coll
                used_where = w
                if (coll, w) != (collection, where):
                    log.warning(
                        "retrieval: fell back to collection=%s where=%s "
                        "(original collection=%s where=%s returned 0)",
                        coll, w, collection, where,
                    )
                break

        chunks = [
            RetrievedChunk(
                conv_id=h["metadata"].get("conv_id", h["id"].split("::")[0]),
                chunk_type=h["metadata"].get("chunk_type", used_collection.replace("conv_", "")),
                text=h["document"],
                score=1.0 - h["distance"],
                metadata={
                    **{k: v for k, v in h["metadata"].items() if k != "chunk_type"},
                    "retrieved_from": used_collection,
                },
            )
            for h in hits
        ]
        return RetrievalResult(chunks=chunks)
