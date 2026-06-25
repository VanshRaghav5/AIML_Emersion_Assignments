from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Optional

from dotenv import load_dotenv

# Auto-load .env at startup (recommended for local dev).
# This does not require reading .env via tools—it's runtime behavior.
load_dotenv()


@dataclass(frozen=True)
class Settings:
    google_api_key: Optional[str] = None
    groq_api_key: Optional[str] = None
    faiss_index_path: str = "vector_store/faiss_index"
    faiss_docs_path: str = "vector_store/faiss_docs.pkl"
    chunk_size: int = 1000
    chunk_overlap: int = 150
    k_retrieval: int = 6


def get_settings() -> Settings:
    google_key = os.getenv("GOOGLE_API_KEY")
    groq_key = os.getenv("GROQ_API_KEY")
    return Settings(
        google_api_key=google_key or None,
        groq_api_key=groq_key or None,
    )
