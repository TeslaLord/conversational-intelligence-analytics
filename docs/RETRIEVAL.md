# Retrieval / agent runtime — what happens when you ask a question

The pipeline ([`docs/PIPELINE.md`](PIPELINE.md)) builds the warehouse and the
vector store offline. This document covers the **online** path: what runs
between you typing a question and seeing an answer.

```
question → Planner (LLM) → Orchestrator (deterministic router)
                                  ↓
                          { SQLTool, RetrievalTool, ConvDetailTool }
                                  ↓
                          Synthesizer (LLM, JSON-only)
                                  ↓
                          AnalystAnswer + trace.jsonl
```

There is **no open-ended ReAct loop**. The planner picks one of six fixed
intents; the orchestrator runs a fixed tool sequence for that intent. The
synthesizer never invents tools or conv_ids — its evidence is filtered
against the union of ids returned by the tools.

---

## 1. Planner ([`cc_insights/agent/planner.py`](../cc_insights/agent/planner.py))

The planner is an LLM call that classifies the question into:

```json
{
  "intent":   "aggregate | retrieve_examples | deep_dive_single_conv | trajectory | cluster_topics | agent_coaching",
  "filters":  { "<key>": "<value or list>" },
  "conv_id":  "C0012345 or null",
  "rationale": "one sentence"
}
```

### Why `conv_id` is almost always `null` — and that is correct

`conv_id` is **only** filled when the user explicitly names a conversation
(e.g. *"Summarize C0050175"*, *"Why did C0001234 escalate?"*). For
ranking/topic/coaching questions there is no single conversation to look at,
so `null` is the right answer. It is not a bug — `null` here means "the
orchestrator should pick the conversations itself via SQL or vector search".

The only intent that *requires* a non-null conv_id is `deep_dive_single_conv`;
`trajectory` falls back to a 1-result vector search if conv_id is null.

### Filter values come from the live data, not the LLM

This is the change that fixed most of the "no results found" failures.

At startup ([`app.py`](../cc_insights/app.py) `_load_enum_values`) we read
`SELECT DISTINCT <col>` from the warehouse for the low-cardinality columns
(`industry`, `product`, `primary_intent`, `outcome`, `language`, `channel`,
`overall_sentiment`) and inject the list into the planner prompt:

```
Allowed values for enum filter keys (USE THESE EXACT STRINGS):
  product: ["4G Pack", "5G Upgrade", "API", "Analytics", ...]
  primary_intent: ["activate_esim", "api_rate_limit", ..., "update_kYC"]
  ...
```

Then `Planner._sanitize_filters` drops any value the LLM returned that is not
in the enum (logged as a warning). This is the safety net for *"5G"* →
filter dropped → semantic search runs unfiltered → hits the real
`product="5G Upgrade"` data.

The planner prompt also tells the LLM to **prefer no filter** for vague
topical phrases, e.g. *"flight issues"* should not become
`product=["Flight"]` because the user might mean booking, cancellation, or
reschedule — let the vector similarity decide.

---

## 2. Orchestrator ([`cc_insights/agent/orchestrator.py`](../cc_insights/agent/orchestrator.py))

For each intent, a fixed tool plan:

| intent                  | tools run, in order                                            |
| ----------------------- | -------------------------------------------------------------- |
| `aggregate`             | `sql.intent_success(filters)` → `retrieval.search(question, filters)` |
| `retrieve_examples`     | `retrieval.search(question, filters)`                          |
| `deep_dive_single_conv` | `conv_detail.get(plan.conv_id)`                                |
| `trajectory`            | (if no conv_id: `retrieval.search(k=1)` to find one) → `conv_detail.get(...)` |
| `cluster_topics`        | `sql.topic_frequency(filters)` → `retrieval.search(question, filters)` |
| `agent_coaching`        | `sql.agents_needing_coaching(filters)` → if rows, `retrieval.search(question, filters + {agent_hash: worst_agent})` |

After tools run, the orchestrator collects every `conv_id` returned (from the
SQL aggregates *and* the retrieval chunks) into `allowed_ids`. This set is
the only universe the synthesizer is allowed to cite.

A JSONL trace per query is written to `data/traces/<query_id>.jsonl`
containing the plan, the tool keys, the allowed_ids list, and the final
answer.

---

## 3. SQLTool ([`cc_insights/agent/tools/sql_tool.py`](../cc_insights/agent/tools/sql_tool.py))

**Strictly parameterized — no free-form text-to-SQL.** Each method builds a
SQL string with `?` placeholders and a typed parameter list, run against the
materialized analytic tables (or `conversations` for the filter-aware
queries):

- `intent_success(filters)` — per `primary_intent`: count, resolution rate,
  avg turns, avg dissatisfaction.
- `top_topics_by_sentiment(sentiment, filters)` — per
  `(industry, product, primary_intent)` for a sentiment slice.
- `domain_intent_success(filters)` — per `(industry, product, intent)`.
- `agents_needing_coaching(filters)` — per `agent_hash`, requires ≥3 convs,
  ordered by `avg_empathy ASC NULLS LAST, avg_dissatisfaction DESC`.
- `topic_frequency(filters)` — per `(intent, industry, product)`, ordered by
  dissatisfaction then count (i.e. "biggest pain points first").

`_build_where(filters)` filters on the whitelisted enum columns only;
unrecognized keys are silently skipped (the planner already sanitized them,
this is belt-and-braces).

Each result includes the raw SQL string in `AggregateResult.sql` so it shows
up in the trace and the UI's "Plan + tools" panel.

---

## 4. RetrievalTool ([`cc_insights/agent/tools/retrieval_tool.py`](../cc_insights/agent/tools/retrieval_tool.py))

Wraps Chroma. One semantic search call, possibly with a metadata
where-filter. The user's question is embedded with the **same embedder used
at indexing time** (`paraphrase-multilingual-MiniLM-L12-v2`, 384-d, cosine).

### Three fallback attempts

```
1. requested collection (default: conv_summary) + filters
2. requested collection + NO filters         ← rescues over-constrained where
3. conv_highlight + NO filters               ← rescues "needle in transcript"
```

Why this matters:

- **Over-constrained where-clauses are the #1 cause of empty results.** Even
  with the planner enum-validation, a perfectly valid filter
  (`outcome="Escalated"`) can starve the result set when combined with the
  vector query. We retry without filters before giving up.
- **Some questions live in the turn text, not the summary.** Asking
  *"customer mentioned biometric glitch"* may not surface from the
  multi-sentence summary but will hit a 1-sentence highlight. The third
  attempt rescues those.

Fallbacks are logged as warnings so the trace tells you the retrieval was
relaxed.

### Where-clause encoding for Chroma

- single key, single value → `{"product": "5G Upgrade"}`
- single key, list of values → `{"product": {"$in": ["5G Upgrade", "Roaming"]}}`
- multiple keys → `{"$and": [{"product": "..."}, {"outcome": "..."}]}`

### Output

`RetrievedChunk(conv_id, chunk_type, text, score, metadata)` where
`score = 1 - cosine_distance` and `metadata` is the per-row metadata we
stored at indexing time (industry, product, intent, sentiment,
dissatisfaction_score, agent_hash, …).

---

## 5. ConvDetailTool ([`cc_insights/agent/tools/conv_detail_tool.py`](../cc_insights/agent/tools/conv_detail_tool.py))

Used by `deep_dive_single_conv` and `trajectory`. Reads one conversation row
plus all its turns (`role`, `clean_text`, `turn_sentiment`,
`turn_sentiment_signed`, `is_rephrase`, `response_latency_min`, …) from
DuckDB and packages them into a `ConvDetail` object that the synthesizer can
narrate turn-by-turn.

---

## 6. Synthesizer ([`cc_insights/agent/synthesizer.py`](../cc_insights/agent/synthesizer.py))

Single LLM call, strict-JSON output, with these guard rails:

1. **Prompt contains the question, the plan, the tool outputs (≤12 kB JSON),
   and the explicit `allowed_conv_ids` list.** The prompt says "every
   evidence.conv_id MUST appear in Allowed conv_ids".
2. **Post-validation drops any evidence whose conv_id is not in `allowed_ids`.**
   If anything was dropped a caveat is appended:
   `"N hallucinated conv_id(s) removed from evidence."`
3. **Confidence is clamped** to `low|medium|high`; any other value → `low`.
4. **If `allowed_ids` is empty** the LLM is instructed to return
   `evidence: []` and lower confidence — no fabrication.

The output is a typed `AnalystAnswer`:
```python
headline: str
body: str
evidence: list[Evidence]   # each cites a real conv_id
caveats: list[str]
confidence: "low" | "medium" | "high"
used_tools: list[str]
plan: QueryPlan
```

---

## Why some queries used to return nothing — a post-mortem

Recorded for future debugging:

| symptom | root cause | fix |
| --- | --- | --- |
| `"5G Upgrade problems"` → empty | Planner emitted `product: ["5G"]`; real value is `"5G Upgrade"`; Chroma where-filter matched zero rows; no fallback existed. | Inject enum values into planner prompt; sanitize unknown values; fall back to no-filter search. |
| `"booking issues"` → empty | Planner invented `product: ["Booking"]` (doesn't exist; closest real value is `primary_intent="book_appointment"`). | Same as above + prompt now says "vague topical phrases belong in the semantic query, not in a hard filter". |
| `"top 5 topics most associated with negative sentiment"` → noisy evidence | SQL gave correct top-5; retrieval pulled some `overall_sentiment=negative` summaries that happened to be off-topic; synthesizer narrated them anyway. | Partially fixed by the retrieval fallback (better hits available now). Long-term: rank evidence by retrieval similarity and trim to top-N before showing. |
| `conv_id: null` in the plan panel | Misunderstood as a bug. It is correct — `conv_id` is only filled when the user names a specific conversation. | Documented above. |

---

## What to check when an answer looks wrong

1. **Open `data/traces/<query_id>.jsonl`** — it contains the plan, the
   `allowed_ids`, and the final answer. Most issues are visible here in
   seconds.
2. **Check the plan filters.** If they look over-constrained or wrong, the
   planner is the culprit; check the warning log for
   `planner: dropped invalid <key> values [...]`.
3. **Check `allowed_ids`.** Empty means retrieval and SQL both returned
   nothing — usually a filter problem.
4. **Watch the `retrieval: fell back to ...` warning.** If it fires often for
   a given filter combination, that filter is too narrow given current data
   volume — either ingest more batches or relax the prompt guidance.
