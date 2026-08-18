"""In-memory FAISS vector search over L2-normalized embeddings."""

import faiss
import numpy as np


class FaissSearchIndex:
    """Wraps a FAISS IndexFlatIP. Knows nothing about SQLite or embeddings."""

    def __init__(self, dimension: int) -> None:
        self._dimension = dimension
        self._index = faiss.IndexFlatIP(dimension)

    def build(self, vectors: np.ndarray) -> None:
        """Replace the index contents with `vectors` (rows already L2-normalized)."""
        self._index = faiss.IndexFlatIP(self._dimension)
        if len(vectors) > 0:
            self._index.add(vectors)

    def search(self, query_vector: np.ndarray, k: int) -> list[tuple[int, float]]:
        """Return up to `k` (row_index, similarity_score) pairs, best first."""
        if self._index.ntotal == 0:
            return []

        k = min(k, self._index.ntotal)
        scores, indices = self._index.search(query_vector.reshape(1, -1), k)
        return [
            (int(idx), float(score))
            for idx, score in zip(indices[0], scores[0], strict=True)
            if idx != -1
        ]
