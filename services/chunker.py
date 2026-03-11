def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> list[str]:
    """
    按字符长度切分文本，并设置重叠区域。
    """
    if not text.strip():
        return []

    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end == text_length:
            break

        start = end - overlap

    return chunks