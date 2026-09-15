"""
Application settings.

Most values come from .env (see .env.example). The embedding model, Ollama
model, and collection name are read from the vector store's own config.json
(written by notebooks/rag_pipeline.ipynb in Phase 2.7) rather than duplicated
here, so retraining the notebook with different settings doesn't require
touching backend code.
"""

import json
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# app/core/config.py -> app/core -> app -> backend/
BACKEND_ROOT = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # Where the persisted Chroma store (copied from the notebook) lives.
    # If given as a relative path (the default), it's resolved relative to
    # the backend/ directory itself -- NOT the current working directory --
    # so this works the same whether you run uvicorn/pytest from backend/,
    # backend/tests/, or the project root.
    vector_store_dir: str = "data/vector_store"

    # Retrieval
    retrieval_k: int = 4

    # CORS: the frontend's origin(s), comma-separated
    cors_allowed_origins: str = "http://localhost:8501"

    # Logging
    log_level: str = "INFO"

    @property
    def vector_store_path(self) -> Path:
        p = Path(self.vector_store_dir)
        if not p.is_absolute():
            p = BACKEND_ROOT / p
        return p

    @property
    def cors_origins_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_allowed_origins.split(",") if origin.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()


@lru_cache
def get_vector_store_config() -> dict:
    """Read the config.json exported by the notebook (Phase 2.7) --
    embedding model name, ollama model name, collection name, chunk settings.
    """
    settings = get_settings()
    config_path = settings.vector_store_path / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(
            f"vector store config not found at {config_path}. "
            "Did you copy data/vector_store/ from the notebook into backend/data/vector_store/?"
        )
    return json.loads(config_path.read_text(encoding="utf-8"))