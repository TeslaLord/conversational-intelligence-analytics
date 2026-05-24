"""Planner: classifies analyst query into intent + filters using the LLM."""
from __future__ import annotations

import json
import logging

from ..models.llm import LLMClient
from ..schemas import QueryPlan


log = logging.getLogger(__name__)


_PLANNER_SYSTEM = (
    "You are the planner for a contact-centre analytics bot. "
    "Classify the analyst's question into ONE intent and extract structured filters. "
    "Respond with strict JSON only."
)

_VALID_FILTER_KEYS = [
    "industry", "product", "primary_intent", "outcome",
    "language", "channel", "overall_sentiment", "overall_urgency",
    "month", "agent_hash",
]

# These keys are low-cardinality enums whose exact values must come from data.
# The planner gets the live value list so it cannot hallucinate (e.g. "5G"
# instead of "5G Upgrade", or "booking" instead of "book_appointment").
_ENUM_KEYS = (
    "industry", "product", "primary_intent", "outcome",
    "language", "channel", "overall_sentiment",
)

_PLANNER_USER_TMPL = """Question: {question}

Valid intents:
- aggregate            : counts / rankings / ratios across conversations.
- retrieve_examples    : "show me examples / similar cases / why".
- deep_dive_single_conv: specific conversation by id; question contains a conv_id.
- trajectory           : how sentiment/urgency changed within a conversation.
- cluster_topics       : recurring themes / concerns / leadership review.
- agent_coaching       : agents needing coaching / empathy / communication.

Valid filter keys (use only these): {keys}

Allowed values for enum filter keys (USE THESE EXACT STRINGS, case-sensitive;
if the user uses a different word, map it to the closest allowed value, or
omit the filter entirely rather than inventing a new value):
{enums}

Guidance:
- conv_id MUST be null unless the question explicitly mentions a conversation
  id like "C0012345".
- Only include a filter if the question clearly constrains that dimension.
  Vague topical phrases ("flight issues", "5G upgrade problems") usually
  belong in the semantic query, NOT in a hard filter -- omitting a filter is
  always safer than guessing one.
- For free-text topics (e.g. "kyc problems") prefer NO filter and let the
  vector search handle it.

Return JSON:
{{
  "intent": "<one of the above>",
  "filters": {{...subset of valid keys, values must be strings or arrays of strings from the allowed list...}},
  "conv_id": "<string or null>",
  "rationale": "<one short sentence>"
}}
"""


class Planner:
    def __init__(
        self,
        llm: LLMClient,
        model: str,
        valid_values: dict[str, list[str]] | None = None,
    ):
        self._llm = llm
        self._model = model
        self._valid_values = valid_values or {}

    def _enum_block(self) -> str:
        if not self._valid_values:
            return "(none provided)"
        lines = []
        for k in _ENUM_KEYS:
            vals = self._valid_values.get(k)
            if vals:
                lines.append(f"  {k}: {json.dumps(vals)}")
        return "\n".join(lines) if lines else "(none provided)"

    def _sanitize_filters(self, filters: dict) -> dict:
        """Drop filter values the planner invented that aren't in the enum."""
        clean: dict = {}
        for k, v in filters.items():
            if k not in _VALID_FILTER_KEYS:
                continue
            allowed = self._valid_values.get(k)
            if not allowed:
                # No enum constraint for this key (e.g. agent_hash, month) -> keep
                clean[k] = v
                continue
            allowed_set = set(allowed)
            if isinstance(v, (list, tuple)):
                kept = [x for x in v if x in allowed_set]
                dropped = [x for x in v if x not in allowed_set]
                if dropped:
                    log.warning("planner: dropped invalid %s values %s", k, dropped)
                if kept:
                    clean[k] = kept
            else:
                if v in allowed_set:
                    clean[k] = v
                else:
                    log.warning("planner: dropped invalid %s value %r", k, v)
        return clean

    def plan(self, question: str) -> QueryPlan:
        user = _PLANNER_USER_TMPL.format(
            question=question,
            keys=_VALID_FILTER_KEYS,
            enums=self._enum_block(),
        )
        data = self._llm.chat_json(self._model, _PLANNER_SYSTEM, user, max_tokens=300)
        intent = data.get("intent")
        if intent not in {
            "aggregate", "retrieve_examples", "deep_dive_single_conv",
            "trajectory", "cluster_topics", "agent_coaching",
        }:
            raise ValueError(f"Planner returned invalid intent: {intent!r}")
        filters = data.get("filters") or {}
        if not isinstance(filters, dict):
            raise ValueError("Planner filters must be an object")
        filters = self._sanitize_filters(filters)
        return QueryPlan(
            intent=intent,
            filters=filters,
            conv_id=data.get("conv_id") or None,
            rationale=str(data.get("rationale", "")),
        )
