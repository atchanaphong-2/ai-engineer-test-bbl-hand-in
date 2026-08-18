"""SQLite-backed persistence for chunk text + embeddings + cache metadata."""

import logging
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1


@dataclass(frozen=True)
class CachedChunks:
    """Chunks and their embeddings as loaded from the cache."""

    texts: list[str]
    embeddings: np.ndarray


class ChunkStore:
    """SQLite-backed repository for chunks, their embeddings, and cache metadata.

    Knows nothing about embedding models or vector search — only how to
    persist and validate the cache in `db_path`.
    """

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path

    def is_valid(self, content_hash: str) -> bool:
        """Check whether the cache exists, is readable, and matches `content_hash`."""
        if not self._db_path.exists():
            return False

        try:
            with self._connect() as conn:
                row = conn.execute(
                    "SELECT content_hash, schema_version FROM meta LIMIT 1"
                ).fetchone()
        except (sqlite3.DatabaseError, sqlite3.OperationalError) as exc:
            logger.warning("Chunk cache at %s is unreadable: %s", self._db_path, exc)
            return False

        if row is None:
            return False

        stored_hash, stored_version = row
        return stored_hash == content_hash and stored_version == SCHEMA_VERSION

    def load(self) -> CachedChunks | None:
        """Load cached chunks and embeddings, or None if the cache can't be read."""
        try:
            with self._connect() as conn:
                rows = conn.execute(
                    "SELECT text, embedding FROM chunks ORDER BY id"
                ).fetchall()
        except (sqlite3.DatabaseError, sqlite3.OperationalError) as exc:
            logger.warning("Failed to load chunk cache from %s: %s", self._db_path, exc)
            return None

        if not rows:
            return None

        texts = [row[0] for row in rows]
        embeddings = np.stack([np.frombuffer(row[1], dtype=np.float32) for row in rows])
        return CachedChunks(texts=texts, embeddings=embeddings)

    def save(self, chunks: list[str], embeddings: np.ndarray, content_hash: str) -> None:
        """Rebuild the cache from scratch with the given chunks/embeddings/hash."""
        with self._connect() as conn:
            conn.execute("DROP TABLE IF EXISTS chunks")
            conn.execute("DROP TABLE IF EXISTS meta")
            conn.execute(
                "CREATE TABLE chunks (id INTEGER PRIMARY KEY, text TEXT NOT NULL, "
                "embedding BLOB NOT NULL)"
            )
            conn.execute(
                "CREATE TABLE meta (content_hash TEXT NOT NULL, "
                "schema_version INTEGER NOT NULL)"
            )
            conn.executemany(
                "INSERT INTO chunks (text, embedding) VALUES (?, ?)",
                [
                    (text, vector.tobytes())
                    for text, vector in zip(chunks, embeddings, strict=True)
                ],
            )
            conn.execute(
                "INSERT INTO meta (content_hash, schema_version) VALUES (?, ?)",
                (content_hash, SCHEMA_VERSION),
            )
            conn.commit()
        logger.info("Saved %d chunks to cache at %s", len(chunks), self._db_path)

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self._db_path)
