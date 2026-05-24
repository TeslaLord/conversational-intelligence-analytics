"""LLM tagging: per-conversation summary + topic_tags + per-agent-turn empathy.

Outputs are cached on disk by the LLM client so re-runs are cheap.
"""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from tqdm import tqdm

from ..models.llm import LLMClient, LLMError


_SUMMARY_SYSTEM = (
    "You analyze customer-support conversations for a contact-centre BI tool. "
    "You always respond with strict JSON."
)
_SUMMARY_USER_TMPL = """Conversation transcript (turn_index | role | text):
{transcript}

Conversation metadata:
- industry: {industry}
- product: {product}
- primary_intent: {primary_intent}
- outcome: {outcome}

Produce strict JSON with keys:
  "summary": one to three sentences, factual, no opinions.
  "topic_tags": list of 1 to 3 short noun-phrase tags (lowercase, snake_case ok).
  "agent_empathy_mean": float 1.0..5.0 averaging empathy/politeness across agent turns
                        (5 = warm, acknowledges feeling, polite; 1 = curt, dismissive).
  "agent_empathy_reason": one short sentence.
"""


def _format_transcript(turns_df: pd.DataFrame, max_turns: int = 24) -> str:
    g = turns_df.sort_values("turn_index").head(max_turns)
    lines = []
    for _, r in g.iterrows():
        # truncate per-turn to keep prompts small
        text = r["clean_text"][:240]
        lines.append(f"{int(r['turn_index'])} | {r['role']} | {text}")
    return "\n".join(lines)


def _tag_one(
    llm: LLMClient,
    model: str,
    conv_id: str,
    turns_df: pd.DataFrame,
    meta: dict,
) -> dict:
    user = _SUMMARY_USER_TMPL.format(
        transcript=_format_transcript(turns_df),
        industry=meta.get("industry"),
        product=meta.get("product"),
        primary_intent=meta.get("primary_intent"),
        outcome=meta.get("outcome"),
    )
    try:
        data = llm.chat_json(model, _SUMMARY_SYSTEM, user, max_tokens=400)
    except LLMError as e:
        return {"conv_id": conv_id, "error": str(e)}
    return {
        "conv_id": conv_id,
        "summary": str(data.get("summary", "")).strip(),
        "topic_tags": [str(t).strip().lower() for t in data.get("topic_tags", [])][:3],
        "agent_empathy_mean": float(data.get("agent_empathy_mean", 3.0)),
        "agent_empathy_reason": str(data.get("agent_empathy_reason", "")).strip(),
    }


def tag_conversations(
    turns: pd.DataFrame,
    conversations: pd.DataFrame,
    llm: LLMClient,
    model: str,
    *,
    max_conversations: int,
    concurrency: int,
) -> pd.DataFrame:
    """Returns conversations DataFrame with summary/topic_tags/empathy filled.
    Only the first `max_conversations` (in stable order) are tagged; the rest
    keep nulls so the user can see exactly what was budget-tagged.
    """
    target = conversations.head(max_conversations).copy()
    results: list[dict] = []
    turns_by_conv = dict(tuple(turns.groupby("conv_id", sort=False)))

    def _job(row):
        meta = row.to_dict()
        cid = meta["conv_id"]
        tdf = turns_by_conv.get(cid)
        if tdf is None or tdf.empty:
            return {"conv_id": cid, "error": "no turns"}
        return _tag_one(llm, model, cid, tdf, meta)

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        future_to_cid = {pool.submit(_job, row): row["conv_id"] for _, row in target.iterrows()}
        pbar = tqdm(as_completed(future_to_cid), total=len(future_to_cid), desc="LLM tagging", unit="conv")
        for f in pbar:
            res = f.result()
            results.append(res)
            cid = res.get("conv_id", "?")
            if res.get("error"):
                pbar.write(f"  [tag] {cid} ERROR: {res['error'][:120]}")
            else:
                tags = ",".join(res.get("topic_tags", []))
                pbar.write(f"  [tag] {cid} empathy={res.get('agent_empathy_mean'):.1f} tags=[{tags}]")

    tag_df = pd.DataFrame(results)
    if "error" in tag_df.columns:
        n_err = int(tag_df["error"].notna().sum())
        if n_err:
            print(f"[tagger] {n_err}/{len(tag_df)} tagging failures (will keep nulls).")
    merged = conversations.merge(
        tag_df[["conv_id", "summary", "topic_tags", "agent_empathy_mean"]]
        if "summary" in tag_df.columns else tag_df[["conv_id"]],
        on="conv_id",
        how="left",
        suffixes=("", "_llm"),
    )
    # apply llm-derived columns
    for col in ("summary", "topic_tags", "agent_empathy_mean"):
        llm_col = f"{col}_llm"
        if llm_col in merged.columns:
            merged[col] = merged[llm_col].where(merged[llm_col].notna(), merged[col])
            merged = merged.drop(columns=[llm_col])
    # Serialize topic_tags as JSON string for Parquet/DuckDB friendliness
    merged["topic_tags"] = merged["topic_tags"].map(
        lambda v: json.dumps(v) if isinstance(v, list) else (v if isinstance(v, str) else json.dumps([]))
    )
    return merged
