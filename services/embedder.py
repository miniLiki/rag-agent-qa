from functools import lru_cache
from sentence_transformers import SentenceTransformer


EMBED_MODEL_NAME = "BAAI/bge-small-zh-v1.5"


@lru_cache(maxsize=1)
def get_embedder() -> SentenceTransformer:
    return SentenceTransformer(EMBED_MODEL_NAME)


def embed_text(text: str) -> list[float]:
    model = get_embedder()
    vector = model.encode(text, normalize_embeddings=True)
    return vector.tolist()


def embed_texts(texts: list[str]) -> list[list[float]]:
    model = get_embedder()
    vectors = model.encode(texts, normalize_embeddings=True)
    return vectors.tolist()


def embedding_dim() -> int:
    model = get_embedder()
    return model.get_sentence_embedding_dimension()