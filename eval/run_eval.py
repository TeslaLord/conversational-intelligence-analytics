"""Evaluation harness.

For each golden question:
  - run the agent
  - check intent classification vs expected
  - check evidence count >= min_evidence
  - check answer body mentions required keywords
  - ask the judge LLM for a faithfulness/relevance/completeness rubric

Prints a summary table and writes results to data/eval_results.json.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from cc_insights.app import build_orchestrator
from cc_insights.config import get_settings
from cc_insights.models.llm import LLMClient


GOLDEN_PATH = Path(__file__).parent / "golden_questions.json"


_JUDGE_SYSTEM = (
    "You are an impartial evaluator scoring an analyst-facing answer. "
    "Score each dimension 1-5. Respond strict JSON."
)
_JUDGE_USER_TMPL = """Question:
{question}

Answer JSON:
{answer}

Score on:
- faithfulness: Are all claims supported by the evidence and structured tool outputs implied by the answer?
- relevance: Does the answer address the question?
- completeness: Is the answer sufficiently informative (not vague)?
- caveats_quality: Are limitations/uncertainty acknowledged appropriately?

Return JSON: {{"faithfulness": int, "relevance": int, "completeness": int, "caveats_quality": int, "comment": "<short>"}}
"""


def main() -> None:
    s = get_settings()
    orch = build_orchestrator()
    judge = LLMClient(
        api_key=s.LLM_API_KEY,
        base_url=s.LLM_BASE_URL,
        cache_path=s.llm_cache_path,
    )

    golden = json.loads(GOLDEN_PATH.read_text())
    results = []
    for q in golden:
        t0 = time.time()
        try:
            answer, qid, latency = orch.ask(q["question"])
            ok_intent = answer.plan.intent == q["expected_intent"]
            body_low = (answer.headline + " " + answer.body).lower()
            ok_keywords = all(kw.lower() in body_low for kw in q.get("must_mention", []))
            ok_evidence = len(answer.evidence) >= q.get("min_evidence", 0)
            scores = judge.chat_json(
                s.JUDGE_LLM_MODEL,
                _JUDGE_SYSTEM,
                _JUDGE_USER_TMPL.format(
                    question=q["question"],
                    answer=json.dumps(answer.model_dump(), default=str)[:6000],
                ),
                max_tokens=300,
            )
            results.append({
                "id": q["id"],
                "question": q["question"],
                "query_id": qid,
                "ok_intent": ok_intent,
                "ok_keywords": ok_keywords,
                "ok_evidence": ok_evidence,
                "latency_ms": latency,
                "judge": scores,
                "answer": answer.model_dump(),
            })
        except Exception as e:
            results.append({
                "id": q["id"], "question": q["question"], "error": str(e),
                "latency_ms": int((time.time() - t0) * 1000),
            })

    out_path = s.DATA_DIR / "eval_results.json"
    out_path.write_text(json.dumps(results, indent=2, default=str))

    print(f"\n=== Eval summary ({len(results)} questions) ===")
    for r in results:
        if "error" in r:
            print(f"  [{r['id']}] ERROR: {r['error']}")
            continue
        j = r["judge"]
        print(
            f"  [{r['id']}] intent={'Y' if r['ok_intent'] else 'N'} "
            f"kw={'Y' if r['ok_keywords'] else 'N'} ev={'Y' if r['ok_evidence'] else 'N'} "
            f"faith={j.get('faithfulness')} rel={j.get('relevance')} "
            f"comp={j.get('completeness')} cav={j.get('caveats_quality')} "
            f"({r['latency_ms']} ms)"
        )
    print(f"\nresults -> {out_path}")


if __name__ == "__main__":
    main()
