"""
Retrieval service.

Loads the persisted Chroma vector store and the sentence-transformers
embedding model ONCE (via FastAPI's lifespan in main.py), and exposes a
retrieve() function used by generation.py.

Mirrors the retrieval logic from notebooks/rag_pipeline.ipynb section 2.4,
kept in sync via the same embedding model name and collection name read
from the vector store's config.json.
"""

import logging

import chromadb
from sentence_transformers import SentenceTransformer

from app.core.config import get_settings, get_vector_store_config

logger = logging.getLogger(__name__)


class RetrievalService:
    def __init__(self) -> None:
        self._client: chromadb.ClientAPI | None = None
        self._collection = None
        self._embedder: SentenceTransformer | None = None
        self._config: dict = {}

    def load(self) -> None:
        """Called once at app startup (FastAPI lifespan)."""
        settings = get_settings()
        self._config = get_vector_store_config()

        logger.info("Loading embedding model: %s", self._config["embedding_model"])
        self._embedder = SentenceTransformer(self._config["embedding_model"])

        logger.info("Loading Chroma vector store from: %s", settings.vector_store_path)
        self._client = chromadb.PersistentClient(path=str(settings.vector_store_path))
        self._collection = self._client.get_collection(self._config["collection_name"])

        logger.info(
            "Vector store loaded: %d chunks in collection '%s'",
            self._collection.count(),
            self._config["collection_name"],
        )

    @property
    def is_loaded(self) -> bool:
        return self._collection is not None

    @property
    def total_chunks(self) -> int:
        if not self.is_loaded:
            return 0
        return self._collection.count()

    @property
    def config(self) -> dict:
        return self._config

    def retrieve(self, query: str, k: int | None = None) -> list[dict]:
        """Embed the query and return the top-k most similar chunks with metadata."""
        if not self.is_loaded:
            raise RuntimeError("RetrievalService.load() was not called before retrieve()")

        settings = get_settings()
        k = k or settings.retrieval_k

        query_embedding = self._embedder.encode([query], convert_to_numpy=True).tolist()
        results = self._collection.query(query_embeddings=query_embedding, n_results=k)

        hits = []
        for text, meta, dist in zip(
            results["documents"][0], results["metadatas"][0], results["distances"][0]
        ):
            hits.append({"text": text, "metadata": meta, "distance": dist})
        return hits


# module-level singleton, populated by the FastAPI lifespan on startup
retrieval_service = RetrievalService()