from dataclasses import dataclass


@dataclass
class RetrievalResult:
    chunk: str
    score: float
    index: int
    source: str