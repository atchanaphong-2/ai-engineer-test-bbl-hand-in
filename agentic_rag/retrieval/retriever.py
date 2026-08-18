"""Facade composing EmbeddingModel + ChunkStore + FaissSearchIndex."""

import asyncio
import hashlib
import logging
import re
from pathlib import Path

from agentic_rag.retrieval.embedding import EmbeddingModel
from agentic_rag.retrieval.index import FaissSearchIndex
from agentic_rag.retrieval.store import ChunkStore

logger = logging.getLogger(__name__)


class KnowledgeBaseRetriever:
    """Local semantic search over a plain-text knowledge base.

    Owns the load-or-rebuild cache lifecycle. This is the only class the
    rest of the app talks to for retrieval.
    """

    def __init__(
        self,
        kb_path: Path,
        store: ChunkStore,
        embedder: EmbeddingModel,
        index: FaissSearchIndex,
        similarity_floor: float,
    ) -> None:
        self._kb_path = kb_path
        self._store = store
        self._embedder = embedder
        self._index = index
        self._similarity_floor = similarity_floor
        self._chunks: list[str] = []
        self._load_or_rebuild()

    def __len__(self) -> int:
        return len(self._chunks)

    async def search(self, query: str, k: int) -> list[str]:
        """Return up to `k` semantically relevant raw text chunks for `query`.

        Runs the embed+FAISS lookup in a thread so it doesn't block the
        event loop — this lets the Data Retriever agent issue several
        concurrent `search_knowledge_base` calls (e.g. one per keyword/
        phrasing) for a multi-part request. Returns an empty list if the
        best match is below `similarity_floor`, rather than forcing a
        low-confidence chunk into the report.
        """
        if not self._chunks:
            return []

        hits = await asyncio.to_thread(self._search_sync, query, k)
        if not hits or hits[0][1] < self._similarity_floor:
            return []

        return [self._chunks[idx] for idx, _score in hits]

    def _search_sync(self, query: str, k: int) -> list[tuple[int, float]]:
        query_vector = self._embedder.encode([query])[0]
        return self._index.search(query_vector, k)

    def _load_or_rebuild(self) -> None:
        if not self._kb_path.exists():
            raise FileNotFoundError(
                f"Knowledge base file not found: {self._kb_path}. "
                "Create it before starting the retriever."
            )

        raw_text = self._kb_path.read_text(encoding="utf-8")
        chunks = self._chunk_text(raw_text)
        content_hash = self._hash(raw_text)

        if self._store.is_valid(content_hash):
            cached = self._store.load()
            if cached is not None:
                self._chunks = cached.texts
                self._index.build(cached.embeddings)
                logger.info("Loaded %d chunks from cache", len(cached.texts))
                return

        logger.info("Rebuilding chunk cache (%d chunks)", len(chunks))
        embeddings = self._embedder.encode(chunks)
        self._index.build(embeddings)
        self._store.save(chunks, embeddings, content_hash)
        self._chunks = chunks

    @staticmethod
    def _chunk_text(raw_text: str) -> list[str]:
        paragraphs = re.split(r"\n\s*\n", raw_text.strip())
        return [p.strip() for p in paragraphs if p.strip()]

    @staticmethod
    def _hash(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()
