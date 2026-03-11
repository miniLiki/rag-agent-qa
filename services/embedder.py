import numpy as np


def simple_embed(text: str, dim: int = 128) -> np.ndarray:
    """
    一个教学用的简化 embedding 方法：
    将文本中字符的 unicode 值做简单映射，生成固定长度向量。
    这个方法不适合正式项目，但适合先打通检索流程。
    """
    vector = np.zeros(dim, dtype=np.float32)

    if not text:
        return vector

    for i, ch in enumerate(text):
        vector[i % dim] += ord(ch) * 0.001

    norm = np.linalg.norm(vector)
    if norm > 0:
        vector = vector / norm

    return vector