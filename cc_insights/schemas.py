"""Pydantic schemas shared across tools, agent, API."""
from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

Sentiment = Literal["negative", "neutral", "positive"]
Urgency = Literal["low", "medium", "high"]


# --- Planner output ---
class QueryPlan(BaseModel):
    intent: Literal[
        "aggregate",
        "retrieve_examples",
        "deep_dive_single_conv",
        "trajectory",
        "cluster_topics",
        "agent_coaching",
    ]
    filters: dict = Field(default_factory=dict)
    conv_id: str | None = None
    rationale: str


# --- Tool outputs ---
class AggregateRow(BaseModel):
    group: dict
    metrics: dict


class AggregateResult(BaseModel):
    sql: str
    rows: list[AggregateRow]
    row_count: int


class RetrievedChunk(BaseModel):
    conv_id: str
    chunk_type: Literal["summary", "window", "highlight"]
    text: str
    score: float
    metadata: dict


class RetrievalResult(BaseModel):
    chunks: list[RetrievedChunk]


class TurnView(BaseModel):
    turn_index: int
    role: str
    text: str
    sentiment: Sentiment | None
    sentiment_score: float | None
    timestamp: datetime | None


class ConvDetail(BaseModel):
    conv_id: str
    industry: str | None
    product: str | None
    primary_intent: str | None
    outcome: str | None
    overall_sentiment: Sentiment | None
    overall_urgency: Urgency | None
    turn_count: int
    duration_minutes: float | None
    is_time_clean: bool
    sentiment_trajectory: list[float]
    summary: str | None
    topic_tags: list[str]
    turns: list[TurnView]


# --- Final analyst answer ---
class Evidence(BaseModel):
    conv_id: str
    snippet: str
    reason: str
    # Hydrated by the orchestrator after synthesis (not produced by the LLM):
    chunk_type: Literal["summary", "window", "highlight"] | None = None
    retrieved_from: str | None = None  # chroma collection name
    turns: list[TurnView] = Field(default_factory=list)


class AnalystAnswer(BaseModel):
    headline: str
    body: str
    evidence: list[Evidence]
    caveats: list[str]
    confidence: Literal["low", "medium", "high"]
    used_tools: list[str]
    plan: QueryPlan


# --- API ---
class AskRequest(BaseModel):
    question: str
    session_id: str | None = None


class AskResponse(BaseModel):
    answer: AnalystAnswer
    latency_ms: int
    query_id: str
