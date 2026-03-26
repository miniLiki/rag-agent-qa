import os
from pymilvus import MilvusClient
from services.embedder import embed_text, embed_texts, embedding_dim
from services.retrieval_types import RetrievalResult


class MilvusVectorStore:
    def __init__(self, db_path: str = "./milvus_rag.db", collection_name: str = "rag_chunks"):
        self.db_path = db_path
        self.collection_name = collection_name
        self.client = MilvusClient(uri=db_path)
        self.dim = None
        self.chunks: list[str] = []

    def _ensure_collection(self):
        print("[MILVUS] ensure_collection start")
        if self.client.has_collection(self.collection_name):
            print("[MILVUS] collection already exists")
            return

        if self.dim is None:
            print("[MILVUS] loading embedding_dim...")
            self.dim = embedding_dim()
            print(f"[MILVUS] embedding dim = {self.dim}")

        print("[MILVUS] creating collection...")
        self.client.create_collection(
            collection_name=self.collection_name,
            dimension=self.dim,
            metric_type="COSINE",
            consistency_level="Strong",
        )
        print("[MILVUS] create_collection done")

    def rebuild(self, chunks: list[str]):
        print("[MILVUS] rebuild start")
        self.chunks = chunks

        if self.dim is None:
            print("[MILVUS] loading embedding_dim in rebuild...")
            self.dim = embedding_dim()
            print(f"[MILVUS] embedding dim = {self.dim}")

        if self.client.has_collection(self.collection_name):
            print("[MILVUS] dropping old collection...")
            self.client.drop_collection(self.collection_name)
            print("[MILVUS] drop done")

        print("[MILVUS] creating collection...")
        self.client.create_collection(
            collection_name=self.collection_name,
            dimension=self.dim,
            metric_type="COSINE",
            consistency_level="Strong",
        )
        print("[MILVUS] create_collection done")

        print("[MILVUS] embedding chunks...")
        vectors = embed_texts(chunks)
        print("[MILVUS] embed_texts done")

        data = []
        for i, (chunk, vector) in enumerate(zip(chunks, vectors)):
            data.append({
                "id": i,
                "vector": vector,
                "text": chunk,
            })

        print("[MILVUS] inserting data...")
        self.client.insert(
            collection_name=self.collection_name,
            data=data,
        )
        print("[MILVUS] insert done")

    def search(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        if not self.chunks:
            return []

        print("[MILVUS] embedding query...")
        query_vector = embed_text(query)
        print("[MILVUS] query embedded")

        results = self.client.search(
            collection_name=self.collection_name,
            data=[query_vector],
            limit=top_k,
            output_fields=["text"],
        )

        retrievals = []
        for item in results[0]:
            entity = item["entity"]
            idx = int(item["id"])
            retrievals.append(
                RetrievalResult(
                    chunk=entity["text"],
                    score=float(item["distance"]),
                    index=idx,
                    source="dense",
                )
            )
        return retrievals