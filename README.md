# Contact-Centre Insights Bot

Analyst-facing bot over a synthetic customer-support conversations dataset.
Combines **DuckDB analytics + Chroma RAG**, orchestrated by a deterministic
planner / tool agent. Ships a Streamlit UI, a FastAPI surface, an offline
build pipeline, and a golden-question evaluation harness.

---

## 1. Layout

```
cc_insights/
  config.py          # pydantic-settings; fails loudly if env vars are missing
  schemas.py         # Pydantic models for every tool I/O contract
  data/              # csv -> parquet, cleaning, PII hashing, stratified subset
  features/          # per-turn + per-conv feature engineering, LLM tagging
  models/            # embeddings, sentiment, OpenAI-compatible LLM client
  storage/           # duckdb warehouse, chroma vector store, sqlite sessions
  agent/             # planner + tools + synthesizer (deterministic)
  pipeline.py        # end-to-end offline build (idempotent, batched)
  api.py             # FastAPI
  ui.py              # Streamlit
eval/
  golden_questions.json
  run_eval.py        # writes ./data/eval_results.json
docs/
  PIPELINE.md  RETRIEVAL.md
scripts/
  run_batches.sh
```

---

## 2. Dataset

The pipeline expects a single CSV named **`customer_support_data.csv`** in the
project root (the same folder as this README). Each row is one turn:

| column | notes |
|---|---|
| `conversation_id` | groups turns |
| `turn_index` | authoritative ordering (timestamps are NOT monotonic) |
| `speaker` | `customer` / `agent` |
| `text` | utterance |
| `timestamp` | ISO 8601, used only for duration features after sanity checks |
| `language` | ISO code |
| `overall_sentiment`, `primary_intent`, `outcome` | silver labels (anchors) |

PII is hashed on ingest using `PII_HASH_SALT` before anything leaves the box.

### How to obtain it

1. **If you have the file already** (e.g. provided with the assignment), drop
   it in the project root as `customer_support_data.csv`. Done.
2. **From the original source** (synthetic Syncora-style schema), export to
   CSV with the columns above. The pipeline only needs the columns listed,
   extra columns are ignored.

`CSV_PATH` and `DATA_DIR` in `.env` may be **absolute or relative**. Relative
paths resolve against the project root, so the defaults in `.env.example`
work as-is for a fresh clone.

---

## 3. Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# then edit .env and set LLM_API_KEY (and PII_HASH_SALT to any random string)
```

### Required environment variables (all of them — no defaults)

| var | purpose |
|---|---|
| `CSV_PATH` | path to `customer_support_data.csv` (relative OK) |
| `DATA_DIR` | where parquet, duckdb, chroma, traces, sqlite live (relative OK) |
| `SUBSET_SIZE` | stratified subset size for tagging/eval |
| `RANDOM_SEED` | reproducibility |
| `EMBEDDING_MODEL` | HF sentence-transformer id |
| `SENTIMENT_MODEL` | HF seq-classification id (XLM-R multilingual default) |
| `LLM_API_KEY` | any OpenAI-compatible key |
| `LLM_BASE_URL` | OpenAI / OpenRouter / vLLM / Ollama endpoint |
| `TAGGING_LLM_MODEL` | model used during offline tagging |
| `SYNTH_LLM_MODEL` | model used by the synthesizer at query time |
| `JUDGE_LLM_MODEL` | model used by `eval/run_eval.py` |
| `TAGGING_MAX_CONVERSATIONS`, `TAGGING_CONCURRENCY` | tagging budget |
| `TOP_K_DENSE`, `TOP_K_FINAL` | retrieval params |
| `PII_HASH_SALT` | any non-empty random string |

### External services

- **LLM provider** — any OpenAI-compatible endpoint (`LLM_BASE_URL`). Default
  example targets `https://api.openai.com/v1` with `gpt-4o-mini`.
- **HuggingFace Hub** — first run downloads the embedding model
  (~120 MB) and the sentiment model (~1.1 GB) into `~/.cache/huggingface`.
  No HF token needed for the public defaults.
- No other network calls. DuckDB, Chroma, and the SQLite session store all
  run locally in `DATA_DIR`.

---

## 4. Run

```bash
# Build everything: clean -> features -> sentiment -> LLM tag -> duckdb -> chroma
# Idempotent: re-running resumes from processed_convs.txt
python -m cc_insights.pipeline --batch-size 100

# Start the analyst UI (chatbot + tool traces)
streamlit run cc_insights/ui.py

# Or the API
uvicorn cc_insights.api:app --reload
# POST /ask {"question": "...", "session_id": "..."}

# Evaluate against the golden set -> ./data/eval_results.json
python -m eval.run_eval
```

The pipeline is batched and resumable — kill it any time, re-run, it picks up
from `data/processed_convs.txt`.

---

## 5. Sample requests & outputs

The agent classifies each question into an **intent** (`aggregate`,
`retrieve`, `conversation_detail`) and dispatches to the matching tool. Every
answer is a structured `AnswerResponse` (see `cc_insights/schemas.py`) with
`headline`, `body`, `evidence[]`, `caveats[]`, and the full `plan` + tool
trace.

### Aggregate (SQL over DuckDB)

> **Q:** "What are the top 5 topics most associated with negative customer sentiment?"

```json
{
  "headline": "Billing, account access, and shipping drive the most negative customer sentiment.",
  "body": "Across 1,000 tagged conversations, the topics with the highest share of negative overall_sentiment are billing (38%), account_access (31%), shipping (27%), refunds (24%), and technical_issue (22%). ...",
  "evidence": [
    {"conv_id": "c_8af…", "snippet": "I have been charged twice this month and no one can…", "score": 0.81},
    ...
  ],
  "caveats": ["Counts limited to the 1,000-conversation tagged subset."],
  "plan": {"intent": "aggregate", "tool": "sql"}
}
```

### Retrieval (Chroma RAG)

> **Q:** "Show me examples where the agent failed to de-escalate an angry customer."

Returns 5 high-friction turn windows with hashed IDs, customer-sentiment
trajectory, and snippets — `plan.intent = "retrieve"`, `plan.tool = "retrieval"`.

### Conversation detail

> **Q:** "Summarize conversation c_8af2…"

Loads the full conversation, runs the synthesizer with the per-conv features
(duration, sentiment trajectory, intent, outcome, agent-response latency
percentiles) — `plan.intent = "conversation_detail"`.

More examples are in `eval/golden_questions.json`.

---

## 6. Evaluation

`python -m eval.run_eval` runs each golden question end-to-end and writes
`./data/eval_results.json`. For every question it records:

- **Intent classification** correctness (`ok_intent`)
- **Keyword coverage** in the answer body (`ok_keywords`)
- **Evidence count** ≥ `min_evidence` (`ok_evidence`)
- **LLM-judge rubric** (1–5): faithfulness, relevance, completeness, caveats quality
- **Latency** per question in ms

The judge prompt is in `eval/run_eval.py`. Using the same provider for tagger
and judge has a known self-preference bias — swap `JUDGE_LLM_MODEL` to a
stronger model for stricter evaluation.

---

## 7. Design notes

- **No default config values.** Missing env vars raise on startup — the
  pipeline never silently runs with the wrong settings.
- **Deterministic planner**, not open ReAct. Tool I/O is Pydantic-typed so
  the synthesizer always sees structured evidence.
- **Hashed agent/customer names** via salted SHA-256 on ingest.
- **Silver labels** from the dataset (`overall_sentiment`, `primary_intent`,
  `outcome`) are kept as anchors; we *augment* per-turn signals with a
  multilingual sentiment classifier and an LLM tagger.
- **Timestamps within `conv_id` are non-monotonic** — ordering is always by
  `turn_index`; duration/latency features are flagged with `is_time_clean`.
- **Triple-indexed vectors** in Chroma: conv summary, 4-turn windows, and
  high-friction turns — chosen at retrieval time by intent.
- **JSONL traces** in `DATA_DIR/traces/` for every analyst query for audit.
- **Persistent LLM cache** in `DATA_DIR/llm_cache.sqlite` so re-running eval
  or repeat questions is free.

---

## 8. Limitations & next steps

- Self-preference risk in the LLM judge (single-provider setup).
- LLM tagging budget capped at `TAGGING_MAX_CONVERSATIONS`; remaining
  conversations rely only on silver labels + classifier signals.
- No reranker on retrieval yet — `TOP_K_FINAL` is a simple top-N cut from
  the dense scorer.
- Streamlit UI is intentionally minimal (single-turn focus, session memory
  via SQLite). Multi-turn follow-up resolution is on the roadmap.
- Latency dominated by the synthesizer LLM call; consider streaming and a
  smaller distilled judge for CI.

---

## 9. Repository

GitHub: <https://github.com/TeslaLord/conversational-intelligence-analytics>
