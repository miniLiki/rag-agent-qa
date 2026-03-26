from services.bm25_store import BM25Store
from services.milvus_store import MilvusVectorStore
from services.retrieval_types import RetrievalResult
from services.reranker import CrossEncoderReranker


class HybridRetriever:
    def __init__(self):
        print("[HYBRID] init start")

        print("[HYBRID] creating dense_store...")
        self.dense_store = MilvusVectorStore()
        print("[HYBRID] dense_store created")

        print("[HYBRID] creating sparse_store...")
        self.sparse_store = BM25Store()
        print("[HYBRID] sparse_store created")

        print("[HYBRID] creating reranker...")
        self.reranker = CrossEncoderReranker()
        print("[HYBRID] reranker created")

        print("[HYBRID] init done")

    def rebuild(self, chunks: list[str]):
        print("[HYBRID] rebuild start")
        self.dense_store.rebuild(chunks)
        print("[HYBRID] dense rebuild done")
        self.sparse_store.rebuild(chunks)
        print("[HYBRID] sparse rebuild done")

    def _rrf_fuse(
        self,
        dense_results: list[RetrievalResult],
        sparse_results: list[RetrievalResult],
        top_k: int,
        k: int = 60,
    ) -> list[RetrievalResult]:
        score_map = {}
        chunk_map = {}

        for rank, item in enumerate(dense_results, start=1):
            score_map[item.index] = score_map.get(item.index, 0.0) + 1.0 / (k + rank)
            chunk_map[item.index] = item

        for rank, item in enumerate(sparse_results, start=1):
            score_map[item.index] = score_map.get(item.index, 0.0) + 1.0 / (k + rank)
            chunk_map[item.index] = item

        fused = []
        for idx, score in score_map.items():
            base = chunk_map[idx]
            fused.append(
                RetrievalResult(
                    chunk=base.chunk,
                    score=float(score),
                    index=idx,
                    source="hybrid_rrf",
                )
            )

        fused.sort(key=lambda x: x.score, reverse=True)
        return fused[:top_k]

    def retrieve(
        self,
        query: str,
        recall_k: int = 8,
        final_k: int = 3,
    ) -> list[RetrievalResult]:
        dense_results = self.dense_store.search(query, top_k=recall_k)
        sparse_results = self.sparse_store.search(query, top_k=recall_k)
        fused = self._rrf_fuse(dense_results, sparse_results, top_k=recall_k)
        reranked = self.reranker.rerank(query, fused, top_k=final_k)
        return reranked