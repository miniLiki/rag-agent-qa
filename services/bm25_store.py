import jieba
from rank_bm25 import BM25Okapi
from services.retrieval_types import RetrievalResult


def tokenize(text: str) -> list[str]:
    return [tok.strip() for tok in jieba.lcut(text) if tok.strip()]


class BM25Store:
    def __init__(self):
        self.chunks: list[str] = []
        self.tokenized_chunks: list[list[str]] = []
        self.bm25 = None

    def rebuild(self, chunks: list[str]):
        self.chunks = chunks
        self.tokenized_chunks = [tokenize(chunk) for chunk in chunks]
        self.bm25 = BM25Okapi(self.tokenized_chunks)

    def search(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        if not self.bm25:
            return []

        tokenized_query = tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]

        return [
            RetrievalResult(
                chunk=self.chunks[idx],
                score=float(score),
                index=int(idx),
                source="bm25",
            )
            for idx, score in ranked
        ]