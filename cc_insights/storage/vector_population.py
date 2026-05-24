"""Build and upsert the three Chroma collections from enriched data.

Collections:
- conv_summary   : one chunk per LLM-tagged conversation summary
- conv_window    : sliding 4-turn windows (stride 3) with mean-pooled embeddings
- conv_highlight : individual customer negative turns + rephrases
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..models.embeddings import Embedder
from .vector_store import VectorStore


def _vec_summary(
    vector: VectorStore, embedder: Embedder, conversations: pd.DataFrame,
) -> None:
    summaries = conversations[
        conversations["summary"].notna() & (conversations["summary"] != "")
    ]
    if summaries.empty:
        return
    docs = summaries["summary"].astype(str).tolist()
    emb = embedder.encode(docs)
    ids = [f"{r['conv_id']}::summary" for _, r in summaries.iterrows()]
    metas = [
        {
            "conv_id": r["conv_id"],
            "chunk_type": "summary",
            "industry": r["industry"],
            "product": r["product"],
            "primary_intent": r["primary_intent"],
            "outcome": r["outcome"],
            "overall_sentiment": r["overall_sentiment"],
            "month": r["month"],
            "agent_hash": r["agent_hash"],
            "language": r["language"],
            "channel": r["channel"],
            "dissatisfaction_score": float(r["dissatisfaction_score"]),
        }
        for _, r in summaries.iterrows()
    ]
    vector.add("conv_summary", ids, emb, docs, metas)


def _vec_windows(
    vector: VectorStore,
    turns: pd.DataFrame,
    conversations: pd.DataFrame,
    turn_embeddings: np.ndarray,
) -> None:
    conv_meta_by_id = conversations.set_index("conv_id")
    ids, docs, metas, embs = [], [], [], []
    turns_reset = turns.reset_index(drop=True)
    for cid, g in turns_reset.groupby("conv_id", sort=False):
        idxs = g.sort_values("turn_index").index.to_list()
        if cid not in conv_meta_by_id.index:
            continue
        cmeta = conv_meta_by_id.loc[cid]
        for start in range(0, len(idxs), 3):
            window = idxs[start : start + 4]
            if not window:
                break
            doc = "\n".join(
                f"{int(turns_reset.at[i, 'turn_index'])} | "
                f"{turns_reset.at[i, 'role']} | {turns_reset.at[i, 'clean_text']}"
                for i in window
            )
            mean_emb = turn_embeddings[window].mean(axis=0)
            n = np.linalg.norm(mean_emb)
            if n > 0:
                mean_emb = mean_emb / n
            ids.append(f"{cid}::w{start}")
            docs.append(doc)
            embs.append(mean_emb)
            metas.append({
                "conv_id": cid,
                "chunk_type": "window",
                "turn_start": int(turns_reset.at[window[0], "turn_index"]),
                "turn_end": int(turns_reset.at[window[-1], "turn_index"]),
                "industry": cmeta["industry"],
                "product": cmeta["product"],
                "primary_intent": cmeta["primary_intent"],
                "outcome": cmeta["outcome"],
                "overall_sentiment": cmeta["overall_sentiment"],
                "month": cmeta["month"],
                "agent_hash": cmeta["agent_hash"],
                "language": cmeta["language"],
            })
    if ids:
        vector.add(
            "conv_window", ids,
            np.vstack(embs).astype("float32"), docs, metas,
        )


def _vec_highlights(
    vector: VectorStore,
    turns: pd.DataFrame,
    conversations: pd.DataFrame,
    turn_embeddings: np.ndarray,
) -> None:
    turns_reset = turns.reset_index(drop=True)
    mask = (
        ((turns_reset["role"] == "customer") & (turns_reset["turn_sentiment"] == "negative"))
        | (turns_reset["is_rephrase"])
    )
    hl = turns_reset[mask]
    if hl.empty:
        return
    conv_meta_by_id = conversations.set_index("conv_id")
    docs = hl["clean_text"].astype(str).tolist()
    embs = turn_embeddings[hl.index.to_numpy()]
    ids = [f"{r['conv_id']}::h{int(r['turn_index'])}" for _, r in hl.iterrows()]
    metas = []
    for _, r in hl.iterrows():
        cid = r["conv_id"]
        cm = conv_meta_by_id.loc[cid] if cid in conv_meta_by_id.index else {}
        get = (lambda k: cm.get(k)) if hasattr(cm, "get") else (lambda k: cm[k])
        metas.append({
            "conv_id": cid,
            "chunk_type": "highlight",
            "turn_index": int(r["turn_index"]),
            "role": r["role"],
            "is_rephrase": bool(r["is_rephrase"]),
            "turn_sentiment": r["turn_sentiment"],
            "industry": get("industry"),
            "product": get("product"),
            "primary_intent": get("primary_intent"),
            "outcome": get("outcome"),
        })
    vector.add("conv_highlight", ids, embs, docs, metas)


def populate_vector_store(
    vector: VectorStore,
    embedder: Embedder,
    turns: pd.DataFrame,
    conversations: pd.DataFrame,
    turn_embeddings: np.ndarray,
) -> None:
    """Upsert summaries, sliding windows, and highlight turns into Chroma."""
    _vec_summary(vector, embedder, conversations)
    _vec_windows(vector, turns, conversations, turn_embeddings)
    _vec_highlights(vector, turns, conversations, turn_embeddings)
