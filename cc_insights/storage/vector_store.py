"""Chroma vector store wrapper.

Three logical collections under one persistent client:
- conv_summary
- conv_window
- conv_highlight
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import chromadb
from chromadb.config import Settings as ChromaSettings


COLLECTIONS = ("conv_summary", "conv_window", "conv_highlight")


def _scrub(meta: dict) -> dict:
    """Chroma metadata values must be str/int/float/bool/None."""
    out: dict = {}
    for k, v in meta.items():
        if v is None or isinstance(v, (str, int, float, bool)):
            out[k] = v
        elif isinstance(v, list):
            out[k] = json.dumps(v)
        else:
            out[k] = str(v)
    return out


class VectorStore:
    def __init__(self, persist_dir: Path):
        persist_dir.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(
            path=str(persist_dir),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._cols = {
            name: self._client.get_or_create_collection(
                name=name, metadata={"hnsw:space": "cosine"}
            )
            for name in COLLECTIONS
        }

    def reset(self) -> None:
        for name in list(self._cols):
            try:
                self._client.delete_collection(name)
            except Exception:
                pass
        self._cols = {
            name: self._client.get_or_create_collection(
                name=name, metadata={"hnsw:space": "cosine"}
            )
            for name in COLLECTIONS
        }

    def add(
        self,
        collection: str,
        ids: list[str],
        embeddings,
        documents: list[str],
        metadatas: list[dict],
    ) -> None:
        if collection not in self._cols:
            raise KeyError(f"Unknown collection: {collection}")
        if not ids:
            return
        self._cols[collection].upsert(
            ids=ids,
            embeddings=embeddings.tolist() if hasattr(embeddings, "tolist") else embeddings,
            documents=documents,
            metadatas=[_scrub(m) for m in metadatas],
        )

    def query(
        self,
        collection: str,
        query_embedding,
        k: int,
        where: dict | None = None,
    ) -> list[dict]:
        col = self._cols[collection]
        res = col.query(
            query_embeddings=[query_embedding.tolist() if hasattr(query_embedding, "tolist") else query_embedding],
            n_results=k,
            where=where or None,
        )
        out: list[dict] = []
        ids = res.get("ids", [[]])[0]
        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        dists = res.get("distances", [[]])[0]
        for i, _id in enumerate(ids):
            out.append({
                "id": _id,
                "document": docs[i] if i < len(docs) else "",
                "metadata": metas[i] if i < len(metas) else {},
                "distance": float(dists[i]) if i < len(dists) else 1.0,
            })
        return out
