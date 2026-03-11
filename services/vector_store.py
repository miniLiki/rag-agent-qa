import faiss
import numpy as np
from services.embedder import simple_embed


class SimpleVectorStore:
    def __init__(self, dim: int = 128):
        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)
        self.chunks = []

    def build_index(self, chunks: list[str]):
        self.chunks = chunks
        vectors = [simple_embed(chunk, self.dim) for chunk in chunks]

        if not vectors:
            raise ValueError("没有可用的 chunks 来建立索引")

        matrix = np.vstack(vectors).astype("float32")
        self.index.add(matrix)

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        if not self.chunks:
            return []

        query_vector = simple_embed(query, self.dim).reshape(1, -1).astype("float32")
        scores, indices = self.index.search(query_vector, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append({
                "chunk": self.chunks[idx],
                "score": float(score),
                "index": int(idx)
            })

        return results