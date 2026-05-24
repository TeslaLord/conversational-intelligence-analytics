"""Offline enrichment pipeline.

Reads raw CSV → cleans + hashes PII → stratified sample of unseen conversations
→ per-turn features (sentiment, embeddings, rephrase, latency) → per-conv
aggregates → LLM tagging (summary, topics, empathy) → persists parquet,
appends to DuckDB warehouse, upserts Chroma vector store.

Incremental by default. Each run picks up the next `SUBSET_SIZE` unseen
conversations; `--reset` wipes state and starts fresh.

Run: python -m cc_insights.pipeline [--reset] [--batch-size N]
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import Settings, get_settings
from .data.load import CLEANING_VERSION, load_raw_csv, prepare_turns
from .data.subset import stratified_conv_sample
from .features.conversation_features import build_conversations
from .features.llm_tagger import tag_conversations
from .features.turn_features import add_turn_features
from .models.embeddings import Embedder
from .models.llm import LLMClient
from .models.sentiment import SentimentClassifier
from .storage.duckdb_store import Warehouse
from .storage.progress_log import (
    append_processed,
    load_pipeline_state,
    load_processed,
    save_pipeline_state,
)
from .storage.vector_population import populate_vector_store
from .storage.vector_store import VectorStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("pipeline")


# =====================================================================
# Shared run context (carries artifacts between stages)
# =====================================================================


@dataclass
class RunContext:
    settings: Settings
    batch_size: int
    processed: set[str] = field(default_factory=set)
    batch_index: int = 0
    total_convs_in_csv: int = 0
    turns_raw: pd.DataFrame | None = None
    turns_enriched: pd.DataFrame | None = None
    turn_embeddings: np.ndarray | None = None
    conversations: pd.DataFrame | None = None
    batch_conv_ids: list[str] = field(default_factory=list)
    embedder: Embedder | None = None


# =====================================================================
# Stage 0: reset state (optional)
# =====================================================================


def stage_reset(cfg: Settings) -> None:
    log.warning("--reset: wiping DuckDB, Chroma, parquet, and processed-convs log")
    for p in (cfg.duckdb_path, cfg.processed_convs_path, cfg.pipeline_state_path):
        if p.exists():
            p.unlink()
    for d in (cfg.chroma_dir, cfg.parquet_dir):
        if d.exists():
            shutil.rmtree(d)
    cfg.ensure_dirs()


def _validate_pipeline_state(cfg: Settings, processed: set[str]) -> None:
    if not processed:
        return
    state = load_pipeline_state(cfg.pipeline_state_path)
    saved_cleaning_version = state.get("cleaning_version")
    if saved_cleaning_version == CLEANING_VERSION:
        return
    raise RuntimeError(
        "Existing processed conversations were built with an older or unknown cleaning "
        f"version (found={saved_cleaning_version!r}, current={CLEANING_VERSION!r}). "
        "Re-run the pipeline with --reset so DuckDB and Chroma are rebuilt with the "
        "current cleaning logic."
    )


# =====================================================================
# Stage 1: load raw CSV
# =====================================================================


def stage_load_raw(ctx: RunContext) -> None:
    log.info("Stage 1: loading CSV from %s", ctx.settings.CSV_PATH)
    raw = load_raw_csv(ctx.settings.CSV_PATH)
    log.info("Loaded %d raw rows", len(raw))
    ctx.turns_raw = raw


# =====================================================================
# Stage 2: clean text + hash PII
# =====================================================================


def stage_clean(ctx: RunContext) -> None:
    log.info("Stage 2: cleaning text and hashing PII")
    assert ctx.turns_raw is not None
    turns = prepare_turns(ctx.turns_raw, salt=ctx.settings.PII_HASH_SALT)
    ctx.total_convs_in_csv = turns["conv_id"].nunique()
    remaining = ctx.total_convs_in_csv - len(ctx.processed)
    log.info(
        "Clean turns: %d, conversations: %d (already processed: %d, remaining: %d)",
        len(turns), ctx.total_convs_in_csv, len(ctx.processed), remaining,
    )
    ctx.turns_raw = turns  # reused as candidate pool in the next stage


# =====================================================================
# Stage 3: stratified subset of unseen conversations
# =====================================================================


def stage_sample(ctx: RunContext) -> bool:
    """Returns True if at least one conversation was sampled."""
    log.info(
        "Stage 3: stratified sample (target=%d, excluding %d already-processed)",
        ctx.batch_size, len(ctx.processed),
    )
    assert ctx.turns_raw is not None
    sampled = stratified_conv_sample(
        ctx.turns_raw,
        n=ctx.batch_size,
        seed=ctx.settings.RANDOM_SEED + len(ctx.processed),
        exclude=ctx.processed,
    )
    if sampled.empty:
        log.info("Sampler returned 0 conversations -- nothing to do.")
        return False
    ctx.batch_conv_ids = sampled["conv_id"].unique().tolist()
    ctx.turns_raw = sampled
    log.info("Sampled %d turns across %d conversations",
             len(sampled), len(ctx.batch_conv_ids))
    return True


# =====================================================================
# Stage 4: per-turn features (sentiment, embeddings, rephrase, latency)
# =====================================================================


def stage_turn_features(ctx: RunContext) -> None:
    assert ctx.turns_raw is not None
    log.info("Stage 4: per-turn features on %d turns", len(ctx.turns_raw))
    sentiment = SentimentClassifier(ctx.settings.SENTIMENT_MODEL)
    embedder = Embedder(ctx.settings.EMBEDDING_MODEL)
    enriched, emb = add_turn_features(ctx.turns_raw, sentiment, embedder)
    ctx.turns_enriched = enriched
    ctx.turn_embeddings = emb
    ctx.embedder = embedder  # reused by the vector-store stage


# =====================================================================
# Stage 5: per-conversation aggregates + LLM tagging
# =====================================================================


def stage_conversation_metrics(ctx: RunContext) -> None:
    assert ctx.turns_enriched is not None
    log.info("Stage 5a: building conversation aggregates")
    convs = build_conversations(ctx.turns_enriched)
    log.info(
        "Stage 5b: LLM tagging up to %d conversations with concurrency=%d",
        ctx.settings.TAGGING_MAX_CONVERSATIONS, ctx.settings.TAGGING_CONCURRENCY,
    )
    llm = LLMClient(
        api_key=ctx.settings.LLM_API_KEY,
        base_url=ctx.settings.LLM_BASE_URL,
        cache_path=ctx.settings.llm_cache_path,
    )
    convs = tag_conversations(
        ctx.turns_enriched, convs, llm,
        model=ctx.settings.TAGGING_LLM_MODEL,
        max_conversations=ctx.settings.TAGGING_MAX_CONVERSATIONS,
        concurrency=ctx.settings.TAGGING_CONCURRENCY,
    )
    ctx.conversations = convs
    log.info("Built %d conversation rows (%d tagged)",
             len(convs), int(convs["summary"].notna().sum()))


# =====================================================================
# Stage 6: persist parquet batch + DuckDB upsert
# =====================================================================


def stage_persist_warehouse(ctx: RunContext) -> None:
    assert ctx.turns_enriched is not None and ctx.conversations is not None
    log.info("Stage 6a: writing parquet batch %04d", ctx.batch_index)
    batch_dir = ctx.settings.parquet_dir / "batches"
    batch_dir.mkdir(parents=True, exist_ok=True)
    ctx.turns_enriched.to_parquet(
        batch_dir / f"turns_{ctx.batch_index:04d}.parquet", index=False,
    )
    ctx.conversations.to_parquet(
        batch_dir / f"conversations_{ctx.batch_index:04d}.parquet", index=False,
    )
    log.info("Stage 6b: upserting into DuckDB warehouse %s", ctx.settings.duckdb_path)
    Warehouse(ctx.settings.duckdb_path).upsert(ctx.turns_enriched, ctx.conversations)


# =====================================================================
# Stage 7: populate Chroma vector store
# =====================================================================


def stage_persist_vectors(ctx: RunContext) -> None:
    log.info("Stage 7: populating Chroma vector store")
    assert (
        ctx.turns_enriched is not None
        and ctx.conversations is not None
        and ctx.turn_embeddings is not None
    )
    embedder = ctx.embedder or Embedder(ctx.settings.EMBEDDING_MODEL)
    vector = VectorStore(ctx.settings.chroma_dir)
    populate_vector_store(
        vector, embedder,
        ctx.turns_enriched, ctx.conversations, ctx.turn_embeddings,
    )
    log.info("Chroma upsert complete")


# =====================================================================
# Stage 8: record progress + write manifest
# =====================================================================


def stage_record_progress(ctx: RunContext) -> None:
    assert ctx.conversations is not None and ctx.turns_enriched is not None
    log.info("Stage 8: appending processed conv_ids and writing manifest")
    append_processed(ctx.settings.processed_convs_path, ctx.batch_conv_ids)
    total_processed = len(ctx.processed) + len(ctx.batch_conv_ids)
    manifest = {
        "batch_index": ctx.batch_index,
        "batch_conversations": len(ctx.batch_conv_ids),
        "total_processed_conversations": total_processed,
        "total_conversations_in_csv": int(ctx.total_convs_in_csv),
        "remaining": int(ctx.total_convs_in_csv - total_processed),
        "batch_rows_turns": int(len(ctx.turns_enriched)),
        "tagged_in_batch": int(ctx.conversations["summary"].notna().sum()),
        "embedding_model": ctx.settings.EMBEDDING_MODEL,
        "sentiment_model": ctx.settings.SENTIMENT_MODEL,
        "tagging_model": ctx.settings.TAGGING_LLM_MODEL,
        "cleaning_version": CLEANING_VERSION,
    }
    (ctx.settings.DATA_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))
    save_pipeline_state(
        ctx.settings.pipeline_state_path,
        {"cleaning_version": CLEANING_VERSION},
    )
    log.info("Manifest: %s", manifest)


# =====================================================================
# Debug: dump ctx data
# =====================================================================


def _log_ctx_summary(ctx: RunContext) -> None:
    def _preview(df: pd.DataFrame, name: str, n: int = 2) -> None:
        log.info("  %s: rows=%d cols=%d", name, len(df), df.shape[1])
        log.info("    columns: %s", ", ".join(df.columns))
        with pd.option_context(
            "display.max_columns", 8,
            "display.width", 160,
            "display.max_colwidth", 40,
        ):
            log.info("    head:\n%s", df.head(n).to_string(index=False))

    log.info("ctx summary:")
    log.info(
        "  batch_index=%d  batch_conv_ids[:3]=%s  (batch total=%d)",
        ctx.batch_index, ctx.batch_conv_ids[:3], len(ctx.batch_conv_ids),
    )
    if ctx.turns_enriched is not None:
        _preview(ctx.turns_enriched, "turns_enriched")
    if ctx.turn_embeddings is not None:
        log.info(
            "  turn_embeddings: shape=%s dtype=%s",
            ctx.turn_embeddings.shape, ctx.turn_embeddings.dtype,
        )
    if ctx.conversations is not None:
        _preview(ctx.conversations, "conversations")


# =====================================================================
# Orchestration
# =====================================================================


def run(reset: bool = False, batch_size: int | None = None) -> None:
    cfg = get_settings()
    cfg.ensure_dirs()
    if reset:
        stage_reset(cfg)

    processed = load_processed(cfg.processed_convs_path)
    _validate_pipeline_state(cfg, processed)
    effective_batch = batch_size or cfg.SUBSET_SIZE
    ctx = RunContext(
        settings=cfg,
        batch_size=effective_batch,
        processed=processed,
        batch_index=len(processed) // max(effective_batch, 1),
    )
    log.info("Data dir: %s | batch size: %d | starting batch index: %d",
             cfg.DATA_DIR, effective_batch, ctx.batch_index)

    stage_load_raw(ctx)
    stage_clean(ctx)
    if ctx.total_convs_in_csv - len(ctx.processed) <= 0:
        log.info("All conversations already processed. Use --reset to rebuild.")
        return
    if not stage_sample(ctx):
        return
    stage_turn_features(ctx)
    stage_conversation_metrics(ctx)
    stage_persist_warehouse(ctx)
    stage_persist_vectors(ctx)
    stage_record_progress(ctx)
    _log_ctx_summary(ctx)
    log.info("Pipeline complete.")


def _main() -> None:
    p = argparse.ArgumentParser(
        description="cc-insights offline build (incremental)",
    )
    p.add_argument("--reset", action="store_true",
                   help="wipe DuckDB, Chroma, parquet, processed log and start fresh")
    p.add_argument("--batch-size", type=int, default=None,
                   help="override SUBSET_SIZE for this run only")
    args = p.parse_args()
    run(reset=args.reset, batch_size=args.batch_size)


if __name__ == "__main__":
    _main()
