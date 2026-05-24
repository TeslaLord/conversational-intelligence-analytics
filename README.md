# Contact-Centre Insights Bot

Analyst-facing bot over the Syncora customer-support conversations dataset.
Combines DuckDB analytics + Chroma RAG, orchestrated by a deterministic
planner/tool agent.

## Layout

```
cc_insights/
  config.py          # pydantic-settings; fails loudly if env vars missing
  schemas.py         # Pydantic models for all tool I/O
  data/              # csv -> parquet, cleaning, PII hashing, stratified subset
  features/          # per-turn + per-conv feature engineering
  models/            # embeddings, sentiment, llm clients
  storage/           # duckdb, chroma, sqlite session
  agent/             # planner, tools, synthesizer
  pipeline.py        # end-to-end offline build
  api.py             # FastAPI
  ui.py              # Streamlit
eval/
  golden_questions.json
  run_eval.py
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then fill EVERY value -- no defaults
```

## Run

```bash
# 1. Build everything (cleaning -> features -> LLM tagging -> duckdb -> chroma)
python -m cc_insights.pipeline

# 2. Serve API
uvicorn cc_insights.api:app --reload

# 3. (optional) Streamlit UI
streamlit run cc_insights/ui.py

# 4. Eval
python -m eval.run_eval
```

## Design notes

- **No default config values.** Missing env vars raise on startup.
- **Deterministic planner**, not open ReAct. Tool I/O is Pydantic-typed.
- **Hashed agent/customer names** before anything leaves the box.
- **Silver labels** from the dataset (`overall_sentiment`, `primary_intent`,
  `outcome`) are anchors; we *augment* per-turn signals with classifiers + LLM.
- **Timestamps within `conv_id` are non-monotonic** — order by `turn_index`,
  flag duration/latency as `is_time_clean`.
- **Triple-indexed vectors**: conv summary / 4-turn windows / high-friction turns.
- **JSONL traces** in `data/traces/` for every analyst query.
