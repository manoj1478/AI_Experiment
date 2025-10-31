"""
Vector store management using ChromaDB.
Uses a SentenceTransformer model to produce embeddings and stores vectors in a
Chroma collection with persistence (duckdb+parquet). Returns objects that
expose `page_content` to remain compatible with the rest of the codebase.
"""
from typing import List, Dict, Optional
import os
import uuid
from types import SimpleNamespace

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from langchain_community.embeddings import SentenceTransformerEmbeddings


class _STEmbeddingFn:
    """Simple wrapper to make SentenceTransformer usable as an embedding function
    for chromadb. chromadb expects a callable that accepts List[str] and
    returns List[List[float]].
    """
    def __init__(self, model: SentenceTransformer):
        self.model = model

    def __call__(self, texts: List[str]) -> List[List[float]]:
        emb = self.model.encode(texts, show_progress_bar=False)
        # sentence-transformers returns numpy arrays; ensure Python lists
        return [vec.tolist() for vec in emb]


class VectorStore:
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        persist_directory: Optional[str] = None,
        collection_name: str = "my_collection",
    ):
        """Initialize a Chroma-backed vector store.

        Args:
            model_name: sentence-transformers model name used for embeddings.
            persist_directory: directory to persist the chroma DB (defaults to
                ./data/vector_store or the VECTOR_STORE_PATH env var).
            collection_name: name of the chroma collection to use/create.
        """
        if persist_directory is None:
            persist_directory = os.getenv("VECTOR_STORE_PATH", "./data/vector_store")

        #self.model = SentenceTransformer(model_name)
        self._embedding_fn = SentenceTransformerEmbeddings(
            model_name=model_name
        )
        self._embedding_fn = _STEmbeddingFn(self.model)

        # Use duckdb+parquet for on-disk persistence by default.
        self._client = chromadb.PersistentClient(path=persist_directory)
        #self._client = chromadb.Client(
        #    Settings(chroma_db_impl="duckdb+parquet", persist_directory=persist_directory)
        #)

        # Get or create collection. Pass embedding function so chroma will use
        # it for queries and additions.
        self.collection = self._client.get_or_create_collection(
            name=collection_name,
            embedding_function=self._embedding_fn,
        )

        # Keep a simple local mapping of id -> original document text/metadata
        # (not strictly required because chroma stores metadata, but useful for
        # compatibility and quick access if needed).
        # We will store text as the `documents` field and any provided metadata
        # in chroma's metadata argument.

    def add_documents(self, documents: List[Dict]):
        """Add documents (LangChain-style Document objects or dicts) to Chroma.

        Each document is expected to provide `page_content` and optionally
        `metadata`.
        """
        texts = [getattr(doc, "page_content", doc.get("page_content")) for doc in documents]
        metadatas = [getattr(doc, "metadata", doc.get("metadata", {})) for doc in documents]
        ids = [str(uuid.uuid4()) for _ in documents]

        # Add to chroma collection
        self.collection.add(ids=ids, documents=texts, metadatas=metadatas)

    def similarity_search(self, query: str, k: int = 4) -> List[SimpleNamespace]:
        """Query the Chroma collection and return results as objects with
        `page_content` attribute to keep compatibility with existing code.
        """
        if self.collection.count() == 0:
            return []

        results = self.collection.query(query_texts=[query], n_results=k)

        # chroma returns lists in fields like 'documents' and 'metadatas'
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [{}])[0]

        # Wrap into SimpleNamespace instances with attributes `page_content` and `metadata`
        wrapped = [SimpleNamespace(page_content=doc, metadata=meta) for doc, meta in zip(docs, metas)]
        return wrapped
