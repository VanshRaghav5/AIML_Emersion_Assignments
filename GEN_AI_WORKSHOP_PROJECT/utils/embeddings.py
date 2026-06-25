from __future__ import annotations

from typing import Optional

from langchain_google_genai import GoogleGenerativeAIEmbeddings


def get_embeddings(google_api_key: str) -> GoogleGenerativeAIEmbeddings:
    return GoogleGenerativeAIEmbeddings(
        google_api_key=google_api_key,
        model="models/gemini-embedding-001"
    )

