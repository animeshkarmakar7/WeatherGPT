import hashlib
import math


class BGEM3Embedder:
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
