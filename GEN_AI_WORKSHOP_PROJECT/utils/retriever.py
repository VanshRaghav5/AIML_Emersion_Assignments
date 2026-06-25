from __future__ import annotations

import os
import pickle
from typing import List, Tuple

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .embeddings import get_embeddings
from config import Settings


def split_text(text: str, *, chunk_size: int, chunk_overlap: int) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " "],
    )
    chunks = splitter.split_text(text)
    return [Document(page_content=c) for c in chunks if c.strip()]


def _ensure_dirs(settings: Settings) -> None:
    os.makedirs(os.path.dirname(settings.faiss_index_path), exist_ok=True)
    os.makedirs("vector_store", exist_ok=True)


def build_faiss_index(
    *,
    settings: Settings,
    documents: List[Document],
) -> FAISS:
    embeddings = get_embeddings(settings.google_api_key)
    return FAISS.from_documents(documents, embeddings)


def save_faiss_index(index: FAISS, settings: Settings) -> None:
    _ensure_dirs(settings)
    index.save_local(settings.faiss_index_path)
    # Also save documents for potential future use.
    with open(settings.faiss_docs_path, "wb") as f:
        pickle.dump(index.docstore._dict, f)


def load_faiss_index(settings: Settings) -> FAISS | None:
    if not os.path.exists(settings.faiss_index_path):
        return None
    embeddings = get_embeddings(settings.google_api_key)
    return FAISS.load_local(
        settings.faiss_index_path,
        embeddings,
        allow_dangerous_deserialization=True,
    )


def get_retriever(index: FAISS, *, k: int):
    return index.as_retriever(search_kwargs={"k": k})

