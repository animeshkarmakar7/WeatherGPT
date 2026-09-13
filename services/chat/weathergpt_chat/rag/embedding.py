"""BGE-M3 embedder with two implementations:

- ``MockBGEM3Embedder``: deterministic SHA-256 hash stub used in unit tests
  (no GPU/model download required).
- ``BGEM3Embedder``: real BAAI/BGE-M3 via ``FlagEmbedding``; activated when
  the env-var ``WEATHERGPT_USE_REAL_EMBEDDER=true`` is set.  Falls back to
  the mock implementation if the library or model is unavailable so the
  service can still start in resource-constrained environments.
"""

import hashlib
import logging
import math
import os

logger = logging.getLogger(__name__)


class MockBGEM3Embedder:
    """Deterministic hash-based embedder for unit tests and local dev.

    Produces normalised 1024-dim vectors from SHA-256 word hashes.
    Semantic similarity is not meaningful — only use for testing.
    """

    def __init__(self, vector_dim: int = 1024) -> None:
        self.vector_dim = vector_dim

    def embed_text(self, text: str) -> list[float]:
        vec = [0.0] * self.vector_dim
        words = text.lower().split()
        if not words:
            return vec
        for i, word in enumerate(words):
            h = int(hashlib.sha256(word.encode("utf-8")).hexdigest(), 16)
            idx = h % self.vector_dim
            weight = 1.0 / math.sqrt(i + 1.0)
            vec[idx] += weight
        norm = math.sqrt(sum(x * x for x in vec))
        if norm > 0:
            vec = [x / norm for x in vec]
        return vec

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_text(t) for t in texts]


class _RealBGEM3Embedder:
    """BAAI/BGE-M3 embedder backed by FlagEmbedding.

    Loaded lazily on first call so the process starts even when the model
    files are not yet cached.  Set ``WEATHERGPT_BGE_MODEL_PATH`` to a local
    directory to avoid downloading at startup (recommended for Docker).
    """

    def __init__(self, vector_dim: int = 1024) -> None:
        self.vector_dim = vector_dim
        self._model = None

    def _get_model(self):
        if self._model is None:
            try:
                from FlagEmbedding import BGEM3FlagModel  # type: ignore[import]

                model_name = os.getenv(
                    "WEATHERGPT_BGE_MODEL_PATH", "BAAI/bge-m3"
                )
                use_fp16 = os.getenv("WEATHERGPT_BGE_FP16", "true").lower() == "true"
                logger.info("Loading BGE-M3 model from '%s' (fp16=%s)…", model_name, use_fp16)
                self._model = BGEM3FlagModel(model_name, use_fp16=use_fp16)
                logger.info("BGE-M3 model loaded successfully.")
            except Exception as exc:
                logger.warning(
                    "Failed to load FlagEmbedding / BGE-M3 (%s). "
                    "Falling back to MockBGEM3Embedder.",
                    exc,
                )
                self._model = MockBGEM3Embedder(self.vector_dim)
        return self._model

    def embed_text(self, text: str) -> list[float]:
        model = self._get_model()
        if isinstance(model, MockBGEM3Embedder):
            return model.embed_text(text)
        result = model.encode(
            [text],
            batch_size=1,
            max_length=512,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )
        return result["dense_vecs"][0].tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        model = self._get_model()
        if isinstance(model, MockBGEM3Embedder):
            return model.embed_batch(texts)
        result = model.encode(
            texts,
            batch_size=32,
            max_length=512,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )
        return [v.tolist() for v in result["dense_vecs"]]


def BGEM3Embedder(vector_dim: int = 1024) -> MockBGEM3Embedder | _RealBGEM3Embedder:
    """Factory that returns the correct embedder based on env config.

    Set ``WEATHERGPT_USE_REAL_EMBEDDER=true`` to use the real BGE-M3 model.
    Defaults to ``MockBGEM3Embedder`` for unit tests and lightweight deploys.
    """
    use_real = os.getenv("WEATHERGPT_USE_REAL_EMBEDDER", "false").lower() == "true"
    if use_real:
        return _RealBGEM3Embedder(vector_dim)
    return MockBGEM3Embedder(vector_dim)
