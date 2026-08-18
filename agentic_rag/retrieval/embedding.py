"""Local sentence-transformer embedding, no external API calls."""

import logging

import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class EmbeddingModel:
    """Thin wrapper around a local sentence-transformer model."""

    def __init__(self, model_name: str) -> None:
        logger.info("Loading embedding model %s", model_name)
        self._model = SentenceTransformer(model_name)

    @property
    def dimension(self) -> int:
        """Vector dimensionality produced by this model."""
        return self._model.get_embedding_dimension()

    def encode(self, texts: list[str]) -> np.ndarray:
        """Encode texts into L2-normalized embedding vectors."""
        return self._model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype(np.float32)
