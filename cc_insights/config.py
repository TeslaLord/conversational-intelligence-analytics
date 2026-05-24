"""Application configuration.

All values are REQUIRED. Missing env vars cause `Settings()` to raise.
No defaults are provided on purpose -- per the project's explicit rule.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Project root = the folder containing the `cc_insights/` package (i.e. the
# `test/` directory when this repo is shared). Used to resolve relative paths
# in `.env` so the project is portable regardless of the current working dir.
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _resolve_under_root(v: Path) -> Path:
    """Resolve a path against PROJECT_ROOT if it is relative."""
    p = Path(v)
    if not p.is_absolute():
        p = (PROJECT_ROOT / p).resolve()
    return p


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        # Load .env from the project root, not the (possibly unrelated) CWD.
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="forbid",
        case_sensitive=True,
    )

    # Data
    CSV_PATH: Path
    DATA_DIR: Path
    SUBSET_SIZE: int = Field(gt=0)
    RANDOM_SEED: int

    # Models
    EMBEDDING_MODEL: str
    SENTIMENT_MODEL: str

    # LLM (any OpenAI-compatible endpoint: OpenAI, OpenRouter, vLLM, Ollama, ...)
    LLM_API_KEY: str
    LLM_BASE_URL: str
    TAGGING_LLM_MODEL: str
    SYNTH_LLM_MODEL: str
    JUDGE_LLM_MODEL: str

    # Tagging budget
    TAGGING_MAX_CONVERSATIONS: int = Field(gt=0)
    TAGGING_CONCURRENCY: int = Field(gt=0)

    # Retrieval
    TOP_K_DENSE: int = Field(gt=0)
    TOP_K_FINAL: int = Field(gt=0)

    # PII
    PII_HASH_SALT: str

    @field_validator("CSV_PATH")
    @classmethod
    def _csv_must_exist(cls, v: Path) -> Path:
        v = _resolve_under_root(v)
        if not v.exists():
            raise ValueError(f"CSV_PATH does not exist: {v}")
        return v

    @field_validator("DATA_DIR")
    @classmethod
    def _data_dir_resolve(cls, v: Path) -> Path:
        return _resolve_under_root(v)

    @field_validator("PII_HASH_SALT")
    @classmethod
    def _salt_not_placeholder(cls, v: str) -> str:
        if v.strip() in {"", "change-me-to-a-random-string", "please-replace-with-random-bytes"}:
            raise ValueError("PII_HASH_SALT must be set to a real random string")
        return v

    # --- derived paths (computed, not configured) ---
    @property
    def parquet_dir(self) -> Path:
        return self.DATA_DIR / "parquet"

    @property
    def duckdb_path(self) -> Path:
        return self.DATA_DIR / "warehouse.duckdb"

    @property
    def chroma_dir(self) -> Path:
        return self.DATA_DIR / "chroma"

    @property
    def session_db_path(self) -> Path:
        return self.DATA_DIR / "session.sqlite"

    @property
    def traces_dir(self) -> Path:
        return self.DATA_DIR / "traces"

    @property
    def llm_cache_path(self) -> Path:
        return self.DATA_DIR / "llm_cache.sqlite"

    @property
    def processed_convs_path(self) -> Path:
        """Newline-delimited file of conv_ids that have completed a pipeline run."""
        return self.DATA_DIR / "processed_convs.txt"

    @property
    def pipeline_state_path(self) -> Path:
        return self.DATA_DIR / "pipeline_state.json"

    def ensure_dirs(self) -> None:
        for p in (
            self.DATA_DIR,
            self.parquet_dir,
            self.chroma_dir,
            self.traces_dir,
        ):
            p.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]
