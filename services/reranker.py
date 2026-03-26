from functools import lru_cache
from sentence_transformers import CrossEncoder
from services.retrieval_types import RetrievalResult


RERANK_MODEL_NAME = "BAAI/bge-reranker-base"


@lru_cache(maxsize=1)
def get_reranker() -> CrossEncoder:
    return CrossEncoder(RERANK_MODEL_NAME)


class CrossEncoderReranker:
    def rerank(
        self,
        query: str,
        candidates: list[RetrievalResult],
        top_k: int = 3,
    ) -> list[RetrievalResult]:
        if not candidates:
            return []

        model = get_reranker()
        pairs = [[query, item.chunk] for item in candidates]
        scores = model.predict(pairs)

        rescored = []
        for item, score in zip(candidates, scores):
            rescored.append(
                RetrievalResult(
                    chunk=item.chunk,
                    score=float(score),
                    index=item.index,
                    source=f"{item.source}+rerank",
                )
            )

        rescored.sort(key=lambda x: x.score, reverse=True)
        return rescored[:top_k]