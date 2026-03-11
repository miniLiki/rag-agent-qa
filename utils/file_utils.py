from pathlib import Path


UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


def save_uploaded_file(file_name: str, file_bytes: bytes) -> str:
    file_path = UPLOAD_DIR / file_name
    with open(file_path, "wb") as f:
        f.write(file_bytes)
    return str(file_path)