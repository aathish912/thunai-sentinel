from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR / ".env"


class ConfigError(RuntimeError):
    """Raised when required configuration is missing."""


@dataclass(frozen=True)
class Settings:
    mongodb_uri: str
    mongodb_db_name: str = "attribution_safe_rag"
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    default_workspace_id: str = "thunai-demo"
    default_source_type: str = "local_file"
    vector_search_index_name: str = "evidence_embedding_index"
    vector_search_path: str = "embedding"
    enable_llm_metadata_enrichment: bool = False
    llm_metadata_model: str = "llama3.2"


def load_settings() -> Settings:
    load_dotenv(ENV_PATH if ENV_PATH.exists() else None)

    mongodb_uri = os.getenv("MONGODB_URI")
    if not mongodb_uri:
        raise ConfigError(
            "Missing MONGODB_URI. Copy .env.example to .env and set your MongoDB connection string."
        )

    return Settings(
        mongodb_uri=mongodb_uri,
        mongodb_db_name=os.getenv("MONGODB_DB_NAME", "attribution_safe_rag"),
        embedding_model_name=os.getenv(
            "EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2"
        ),
        default_workspace_id=os.getenv("WORKSPACE_ID", "thunai-demo"),
        default_source_type=os.getenv("SOURCE_TYPE", "local_file"),
        vector_search_index_name=os.getenv(
            "VECTOR_SEARCH_INDEX_NAME", "evidence_embedding_index"
        ),
        vector_search_path=os.getenv("VECTOR_SEARCH_PATH", "embedding"),
        enable_llm_metadata_enrichment=os.getenv(
            "ENABLE_LLM_METADATA_ENRICHMENT", "false"
        ).strip().lower()
        in {"1", "true", "yes", "on"},
        llm_metadata_model=os.getenv("LLM_METADATA_MODEL", "llama3.2"),
    )
