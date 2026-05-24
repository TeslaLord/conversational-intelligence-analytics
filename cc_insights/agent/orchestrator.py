"""Orchestrator: ties planner + tools + synthesizer.

Deterministic routing -- no open-ended ReAct loop. Each plan.intent maps to a
fixed sequence of tool calls. Tool outputs are serialized into the synthesizer
prompt; the synthesizer's evidence is validated against the union of conv_ids
returned by tools.
"""
from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any

from ..schemas import AnalystAnswer
from .planner import Planner
from .synthesizer import Synthesizer
from .tools.conv_detail_tool import ConvDetailTool
from .tools.retrieval_tool import RetrievalTool
from .tools.sql_tool import SQLTool


class Orchestrator:
    def __init__(
        self,
        planner: Planner,
        synthesizer: Synthesizer,
        sql: SQLTool,
        retrieval: RetrievalTool,
        conv_detail: ConvDetailTool,
        traces_dir: Path,
        top_k_final: int,
    ):
        self.planner = planner
        self.synthesizer = synthesizer
        self.sql = sql
        self.retrieval = retrieval
        self.conv_detail = conv_detail
        self.traces_dir = traces_dir
        self.traces_dir.mkdir(parents=True, exist_ok=True)
        self.top_k_final = top_k_final

    def ask(self, question: str) -> tuple[AnalystAnswer, str, int]:
        t0 = time.time()
        query_id = str(uuid.uuid4())
        trace: dict[str, Any] = {"query_id": query_id, "question": question}

        plan = self.planner.plan(question)
        trace["plan"] = plan.model_dump()

        tool_outputs: dict[str, Any] = {}
        used_tools: list[str] = []
        allowed_ids: set[str] = set()
        # conv_id -> {chunk_type, retrieved_from} for the first chunk that
        # surfaced this conversation. Used to hydrate Evidence later.
        chunk_provenance: dict[str, dict[str, str]] = {}

        def _absorb(ret) -> None:
            for c in ret.chunks:
                allowed_ids.add(c.conv_id)
                if c.conv_id not in chunk_provenance:
                    chunk_provenance[c.conv_id] = {
                        "chunk_type": c.chunk_type,
                        "retrieved_from": c.metadata.get(
                            "retrieved_from", f"conv_{c.chunk_type}"
                        ),
                    }

        intent = plan.intent
        filters = plan.filters

        if intent == "aggregate":
            agg = self.sql.intent_success(filters)
            tool_outputs["intent_success"] = agg.model_dump()
            used_tools.append("sql.intent_success")
            # also pull a couple of examples to ground
            ret = self.retrieval.search(question, filters, k=self.top_k_final)
            tool_outputs["examples"] = ret.model_dump()
            used_tools.append("retrieval.search")
            _absorb(ret)

        elif intent == "retrieve_examples":
            ret = self.retrieval.search(question, filters, k=self.top_k_final)
            tool_outputs["examples"] = ret.model_dump()
            used_tools.append("retrieval.search")
            _absorb(ret)

        elif intent == "deep_dive_single_conv":
            if not plan.conv_id:
                raise ValueError("deep_dive_single_conv requires a conv_id in the plan")
            cd = self.conv_detail.get(plan.conv_id)
            tool_outputs["conv_detail"] = cd.model_dump()
            used_tools.append("conv_detail.get")
            allowed_ids.add(cd.conv_id)
            chunk_provenance[cd.conv_id] = {
                "chunk_type": "summary",
                "retrieved_from": "conv_detail (by id)",
            }

        elif intent == "trajectory":
            if not plan.conv_id:
                # fall back: retrieve, pick top, then deep-dive
                ret = self.retrieval.search(question, filters, k=1)
                if not ret.chunks:
                    raise ValueError("trajectory requested but no conversation matched")
                cid = ret.chunks[0].conv_id
                _absorb(ret)
            else:
                cid = plan.conv_id
                chunk_provenance.setdefault(cid, {
                    "chunk_type": "summary",
                    "retrieved_from": "conv_detail (by id)",
                })
            cd = self.conv_detail.get(cid)
            tool_outputs["conv_detail"] = cd.model_dump()
            used_tools.append("conv_detail.get")
            allowed_ids.add(cd.conv_id)

        elif intent == "cluster_topics":
            agg = self.sql.topic_frequency(filters)
            tool_outputs["topic_frequency"] = agg.model_dump()
            used_tools.append("sql.topic_frequency")
            ret = self.retrieval.search(question, filters, k=self.top_k_final)
            tool_outputs["examples"] = ret.model_dump()
            used_tools.append("retrieval.search")
            _absorb(ret)

        elif intent == "agent_coaching":
            agg = self.sql.agents_needing_coaching(filters)
            tool_outputs["agent_scorecard"] = agg.model_dump()
            used_tools.append("sql.agents_needing_coaching")
            # pull examples from worst agent's conversations if any
            if agg.rows:
                worst = agg.rows[0].group.get("agent_hash")
                if worst:
                    ret = self.retrieval.search(
                        question, {**filters, "agent_hash": worst}, k=self.top_k_final
                    )
                    tool_outputs["examples"] = ret.model_dump()
                    used_tools.append("retrieval.search")
                    _absorb(ret)

        else:
            raise ValueError(f"Unhandled intent: {intent}")

        trace["tool_outputs_keys"] = list(tool_outputs.keys())
        trace["allowed_ids"] = sorted(allowed_ids)

        answer = self.synthesizer.synthesize(
            question=question,
            plan=plan,
            tool_outputs=tool_outputs,
            allowed_conv_ids=sorted(allowed_ids),
            used_tools=used_tools,
        )

        # Hydrate every Evidence with provenance + full transcript so the UI
        # can show where the conv was retrieved from and the whole exchange.
        for ev in answer.evidence:
            prov = chunk_provenance.get(ev.conv_id)
            if prov:
                ev.chunk_type = prov["chunk_type"]  # type: ignore[assignment]
                ev.retrieved_from = prov["retrieved_from"]
            try:
                cd = self.conv_detail.get(ev.conv_id)
                ev.turns = cd.turns
            except Exception:
                ev.turns = []

        latency_ms = int((time.time() - t0) * 1000)
        trace["latency_ms"] = latency_ms
        trace["answer"] = answer.model_dump()
        with (self.traces_dir / f"{query_id}.jsonl").open("w") as f:
            f.write(json.dumps(trace, default=str))
        return answer, query_id, latency_ms
