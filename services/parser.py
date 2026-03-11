from pathlib import Path
import fitz


def parse_pdf(file_path: str) -> str:
    text_parts = []
    doc = fitz.open(file_path)
    for page in doc:
        page_text = page.get_text()
        if page_text:
            text_parts.append(page_text)
    doc.close()
    return "\n".join(text_parts).strip()


def parse_txt(file_path: str) -> str:
    return Path(file_path).read_text(encoding="utf-8").strip()


def parse_document(file_path: str) -> str:
    suffix = Path(file_path).suffix.lower()

    if suffix == ".pdf":
        return parse_pdf(file_path)
    elif suffix == ".txt":
        return parse_txt(file_path)
    else:
        raise ValueError(f"暂不支持的文件类型: {suffix}")