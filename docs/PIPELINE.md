# Pipeline stages — what each one actually does

The pipeline runs **incrementally**. Every invocation processes one batch of
unseen conversations, appends to the warehouse, and records the conv_ids in
`data/processed_convs.txt`. Re-running picks up the next batch.

```
stage_reset  →  stage_load_raw  →  stage_clean  →  stage_sample  →
stage_turn_features  →  stage_conversation_metrics  →
stage_persist_warehouse  →  stage_persist_vectors  →  stage_record_progress
```

A `RunContext` dataclass is threaded through every stage. Each stage mutates
the bits it owns; later stages read what earlier ones produced.

---

## 0. `stage_reset` (optional, only with `--reset`)

Deletes `data/warehouse.duckdb`, `data/chroma/`, `data/parquet/`, and
`data/processed_convs.txt`. The LLM cache (`data/llm_cache.sqlite`) is **not**
deleted — re-tagging the same conversations is free.

---

## 1. `stage_load_raw`

Reads the entire CSV into a single pandas DataFrame
(`ctx.turns_raw`). Robust to encoding issues (tries utf-8 then latin1) and
skips malformed lines. Validates that all required columns are present.

This step is in-memory; for a 50 MB CSV (~360k turns) it uses ~250 MB RAM.

---

## 2. `stage_clean`

Three things happen here:

1. **Text cleaning** — splits each turn into sentences; drops any tail
   sentence where >50 % of tokens look like consonant-run gibberish (the
   synthetic padding in this dataset). Always keeps at least the first
   sentence. Caps output at ~400 chars.
2. **PII hashing** — `customer_name` and `agent_name` are replaced with
   16-char SHA-256 hashes salted by `PII_HASH_SALT`. The raw `text` column is
   dropped; only `clean_text` survives. So **no raw PII reaches downstream
   stages or the LLM**.
3. **Type coercion + ordering** — `timestamp` parsed to UTC datetimes,
   `turn_index` to Int64, rows sorted by `(conv_id, turn_index)`. Timestamps
   are **not** trusted for ordering (they are non-monotonic in this dataset);
   `turn_index` is the source of truth.

Output: `ctx.turns_raw` — clean turn-level DataFrame.

---

## 3. `stage_sample`

Stratified by `outcome × primary_intent × language` at the **conversation**
level. Within each stratum it draws proportionally to the stratum's share of
the *remaining* pool, with a floor of 1 per non-empty stratum, then trims to
`batch_size` if it overshot.

### Why stratify?

A purely random sample of N conversations from 64k will under-represent rare
strata (e.g. `Escalated` × `dispute_charge` × `Hinglish`). Stratification
keeps the analytic mix representative even at small N, so aggregates from
batch 1 are roughly comparable to aggregates from batch 32.

### Will running batch=50 then batch=80 produce duplicates? **No.**

Each successful batch appends its conv_ids to `data/processed_convs.txt`. On
the next run, `stage_sample` excludes that set from the candidate pool
*before* stratifying. The random seed is also bumped by `len(processed)` so
each batch draws a different random slice within each stratum.

The two safety nets if anything goes wrong:
- `Warehouse.upsert` does `DELETE WHERE conv_id IN (...)` before inserting,
  so even a re-run of the same conv_ids is idempotent (no duplicate rows).
- Chroma uses `upsert(ids=...)` with deterministic IDs like
  `C0001234::summary` — same ID overwrites in place.

Output: `ctx.turns_raw` (now filtered to this batch), `ctx.batch_conv_ids`.

---

## 4. `stage_turn_features`

Runs two models over every turn and derives per-turn metrics.

### Models
- **Sentiment** — `cardiffnlp/twitter-xlm-roberta-base-sentiment` (multilingual,
  3-class).
- **Embedder** — `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
  (384-d, normalized).

### Columns added to `ctx.turns_enriched`

| column | meaning |
|---|---|
| `turn_sentiment` | label: `negative` / `neutral` / `positive` |
| `turn_sentiment_score` | confidence of the predicted label, 0–1 |
| `turn_sentiment_signed` | `-score` for negative, `+score` for positive, `0` for neutral. Used to compute trajectory math at conversation level (sum, slope, start vs end) |
| `is_rephrase` | customer turn whose embedding cosine-similarity to its previous *customer* turn is > 0.80 → they re-asked the same thing |
| `is_clarifying_q` | agent turn that ends with `?` or contains "could you / can you confirm / can you share / please confirm" |
| `response_latency_min` | gap **in minutes** to the previous opposite-role turn. NaN if either timestamp is missing or the gap is negative (the dataset has many non-monotonic timestamps; we only count clean monotonic gaps) |

`ctx.turn_embeddings` is also kept around — a `(N, 384)` numpy array, one row
per turn, reused later by `stage_persist_vectors` for the window and highlight
collections (we don't re-embed).

---

## 5. `stage_conversation_metrics`

**5a. Pure aggregates** (`build_conversations`) — one row per conv_id with:

- counts: `turn_count`, `customer_turn_count`, `agent_turn_count`, `rephrase_count`
- sentiment: `sentiment_start`, `sentiment_end`, `sentiment_slope`,
  `overall_sentiment` (mode), `negative_turn_ratio`
- timing: `duration_minutes`, `avg_response_latency_min`,
  `p95_response_latency_min`, `is_time_clean` (false when timestamps were too
  broken to trust)
- outcome flags: `task_completed` (1 if CSV outcome == "Resolved"),
  `escalation_requested`
- composite: **`dissatisfaction_score` =
  `0.5 · negative_turn_ratio + 0.2 · min(rephrase_count/3, 1) + 0.3 · (1 − task_completed)`**
- metadata copied from the first turn: industry, product, primary_intent,
  outcome, language, channel, agent_hash, month

**5b. LLM tagging** (`tag_conversations`) — for up to
`TAGGING_MAX_CONVERSATIONS` conversations, sends the (truncated, PII-free)
transcript to the LLM and fills three columns:

| column | what the LLM produces |
|---|---|
| `summary` | 1–3 factual sentences |
| `topic_tags` | 1–3 short snake_case noun-phrases (JSON-stringified for parquet) |
| `agent_empathy_mean` | float 1.0–5.0 averaged across agent turns |

### Why `topic_tags` if `primary_intent` already exists?

`primary_intent` is a **single coarse label from the CSV** (e.g. `login_issue`,
`refund_status`). It is preset by whoever labelled the dataset and is often:
- too generic (`general_inquiry` covers a huge bag),
- mislabelled or stale,
- limited to one label per conversation.

`topic_tags` are **multi-label, finer, and derived from the actual transcript**.
A `login_issue` conversation might get `["kyc_failure", "biometric_glitch"]`,
which lets the agent answer "what are the recurring concerns inside
login_issue?" — a question `primary_intent` alone cannot answer.

All calls are concurrent (`TAGGING_CONCURRENCY`) and the responses are cached
on disk by `(model, prompt_hash)` so re-runs cost nothing.

Output: `ctx.conversations` — one row per conv_id, with summary/tags/empathy
filled (or null for the budget-skipped tail).

---

## 6. `stage_persist_warehouse`

**6a. Parquet** — writes `data/parquet/batches/turns_NNNN.parquet` and
`conversations_NNNN.parquet`. One pair per run. This is the durable
source-of-truth on disk; the warehouse can be rebuilt from these.

**6b. DuckDB upsert** — opens `data/warehouse.duckdb` and:
1. If `turns` / `conversations` tables don't exist yet → creates them.
2. Otherwise deletes any rows whose `conv_id` is in the incoming batch, then
   inserts the new rows. (Idempotent.)
3. Rebuilds the four materialized analytic tables:
   - `intent_stats` — per `primary_intent` resolution rate, turn counts,
     dissatisfaction
   - `domain_intent_stats` — per `(industry, product, primary_intent)`
   - `agent_scorecard` — per `agent_hash` resolution rate, empathy mean,
     sentiment lift, handle-time, dissatisfaction
   - `month_topic_stats` — per `(month, industry, product, intent, sentiment)`
4. Ensures index `idx_turns_conv` on `turns(conv_id)`.

### Correction: is conversation text stored in DuckDB?

**Yes.** The `turns` table includes the `clean_text` column. So DuckDB holds:
- full enriched turn-level data including the cleaned text body, AND
- conversation-level aggregates including the LLM `summary` string.

What it does **not** hold is the raw, pre-cleaning text or any PII — those
were dropped in stage 2.

---

## 7. `stage_persist_vectors`

Populates three Chroma collections under `data/chroma/`. All three use the
**same embedder** as turn features (384-d, cosine).

| collection | one entry per | document text | embedding source |
|---|---|---|---|
| `conv_summary` | conversation (only those with an LLM summary) | the summary string | freshly encoded summary |
| `conv_window` | 4-turn sliding window, stride 3, within each conversation | concatenated `index \| role \| text` of the 4 turns | mean of the 4 turn embeddings (re-normalized), reused from stage 4 |
| `conv_highlight` | individual customer-negative or rephrase turns | the single turn's clean_text | that turn's embedding, reused from stage 4 |

### Correction: are vectors built only from metrics?

**No — vectors are built from actual conversation text.** Specifically:
- summaries (LLM-generated text),
- 4-turn transcript windows (verbatim cleaned text), and
- individual highlight turns (verbatim cleaned text).

The **metadata** attached to each vector entry is the metric side: industry,
product, primary_intent, outcome, sentiment, dissatisfaction_score, agent_hash,
etc. The agent uses that metadata to pre-filter vector search (e.g.
"semantic search only inside `outcome=Escalated, industry=Fintech`").

All upserts use deterministic IDs (`C0001234::summary`, `::w0`, `::h7`), so
re-running the same batch is a no-op.

---

## 8. `stage_record_progress`

- Appends `ctx.batch_conv_ids` to `data/processed_convs.txt`.
- Writes `data/manifest.json` with batch counters, remaining count, and the
  model names used.

The next run reads `processed_convs.txt` first and excludes those conv_ids
from sampling.

---

## What's in `ctx` at the end of a run?

The pipeline now logs a `ctx` summary after stage 8. Look for the
`ctx summary` log line — it dumps shapes, dtypes, and a 2-row preview of
each DataFrame plus the embedding-matrix shape, so you can see exactly what
flowed through.

Example (truncated):

```
ctx summary:
  batch_index=0 batch_conv_ids[:3]=['C0001234', 'C0005678', 'C0009999']
  turns_enriched: rows=4821 cols=21
    columns: conv_id, turn_index, role, clean_text, timestamp, ...
    head:
       conv_id  turn_index    role  clean_text                    turn_sentiment ...
       C00012        0        customer  My card got declined ...     negative
       C00012        1        agent     Sorry to hear that, can ...  neutral
  turn_embeddings: shape=(4821, 384) dtype=float32
  conversations: rows=1000 cols=28
    head:
       conv_id  turn_count  dissatisfaction_score  summary               topic_tags
       C00012   6           0.42                   Customer disputes ... ["card_dispute"]
```
