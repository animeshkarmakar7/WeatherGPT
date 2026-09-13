import logging
import os

logger = logging.getLogger(__name__)


class BGEM3Embedder:
    def __init__(self, model_path: str = "BAAI/bge-m3", use_fp16: bool = False, vector_dim: int = 1024) -> None:
        self.model_path = model_path
        self.use_fp16 = use_fp16
        self.vector_dim = vector_dim
        self._model = None

    def _get_model(self):
        if self._model is None:
            try:
                from FlagEmbedding import BGEM3FlagModel
            except Exception as exc:
                raise RuntimeError("FlagEmbedding is required for production BGE-M3 embeddings") from exc
            model_path = os.getenv("WEATHERGPT_BGE_MODEL_PATH", self.model_path)
            use_fp16 = os.getenv("WEATHERGPT_BGE_FP16", str(self.use_fp16).lower()).lower() == "true"
            logger.info("Loading BGE-M3 from %s", model_path)
            self._model = BGEM3FlagModel(model_path, use_fp16=use_fp16)
        return self._model

    def embed_text(self, text: str) -> list[float]:
        if not text.strip():
            raise ValueError("Cannot embed empty text")
        result = self._get_model().encode(
            [text],
            batch_size=1,
            max_length=8192,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )
        vector = result["dense_vecs"][0].tolist()
        if len(vector) != self.vector_dim:
            raise RuntimeError(f"BGE-M3 returned dimension {len(vector)}, expected {self.vector_dim}")
        return vector

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        if any(not text.strip() for text in texts):
            raise ValueError("Cannot embed empty text")
        result = self._get_model().encode(
            texts,
            batch_size=32,
            max_length=8192,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )
        vectors = [vector.tolist() for vector in result["dense_vecs"]]
        if any(len(vector) != self.vector_dim for vector in vectors):
            raise RuntimeError("BGE-M3 returned an unexpected vector dimension")
        return vectors
