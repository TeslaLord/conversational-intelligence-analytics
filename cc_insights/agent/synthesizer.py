"""Synthesizer: turns tool outputs into a typed AnalystAnswer with citation check."""
from __future__ import annotations

import json
from typing import Any

from ..models.llm import LLMClient
from ..schemas import AnalystAnswer, Evidence, QueryPlan


_SYNTH_SYSTEM = (
    "You are an analyst-facing assistant for contact-centre data. "
    "You must ground every claim in the provided tool outputs and cite real conv_ids only. "
    "If the evidence is insufficient, say so plainly. Respond with strict JSON only."
)

_SYNTH_USER_TMPL = """Analyst question:
{question}

Plan: {plan_json}

Tool outputs (JSON):
{tools_json}

Allowed conv_ids for citation: {allowed_ids}

Return JSON:
{{
  "headline": "<one sentence>",
  "body": "<2-5 sentences explaining the finding, referencing numbers from tool outputs>",
  "evidence": [{{"conv_id": "...", "snippet": "...", "reason": "..."}}],
  "caveats": ["short caveat", ...],
  "confidence": "low|medium|high"
}}
Rules:
- Every evidence.conv_id MUST appear in Allowed conv_ids; do not invent ids.
- If Allowed conv_ids is empty, return evidence: [] and lower the confidence.
- Caveats must mention any silver-label or small-sample concerns.
"""


class Synthesizer:
    def __init__(self, llm: LLMClient, model: str):
        self._llm = llm
        self._model = model

    def synthesize(
        self,
        question: str,
        plan: QueryPlan,
        tool_outputs: dict[str, Any],
        allowed_conv_ids: list[str],
        used_tools: list[str],
    ) -> AnalystAnswer:
        user = _SYNTH_USER_TMPL.format(
            question=question,
            plan_json=plan.model_dump_json(),
            tools_json=json.dumps(tool_outputs, default=str)[:12000],
            allowed_ids=allowed_conv_ids,
        )
        data = self._llm.chat_json(self._model, _SYNTH_SYSTEM, user, max_tokens=900)
        evidence_raw = data.get("evidence") or []
        # Hallucinated id filter
        allowed = set(allowed_conv_ids)
        evidence: list[Evidence] = []
        dropped = 0
        for e in evidence_raw:
            cid = str(e.get("conv_id", ""))
            if cid and cid in allowed:
                evidence.append(Evidence(
                    conv_id=cid,
                    snippet=str(e.get("snippet", ""))[:400],
                    reason=str(e.get("reason", ""))[:200],
                ))
            else:
                dropped += 1
        caveats = [str(c) for c in (data.get("caveats") or [])]
        if dropped:
            caveats.append(f"{dropped} hallucinated conv_id(s) removed from evidence.")
        confidence = data.get("confidence", "low")
        if confidence not in {"low", "medium", "high"}:
            confidence = "low"
        return AnalystAnswer(
            headline=str(data.get("headline", "")).strip() or "(no headline)",
            body=str(data.get("body", "")).strip(),
            evidence=evidence,
            caveats=caveats,
            confidence=confidence,  # type: ignore[arg-type]
            used_tools=used_tools,
            plan=plan,
        )
