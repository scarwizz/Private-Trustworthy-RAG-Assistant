# src/data/loader.py
from pathlib import Path
from typing import Tuple, List, Dict
import pdfplumber

SUPPORTED_TEXT_EXTS = {".txt", ".md", ".markdown"}

def load_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")

def load_pdf_file(path: Path) -> str:
    text_pages = []
    with pdfplumber.open(path) as pdf:
        for i, page in enumerate(pdf.pages):
            t = page.extract_text()
            if t:
                text_pages.append(t)
    return "\n\n".join(text_pages)

def load_document(path: str) -> Tuple[str, Dict]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Document not found: {path}")
    ext = p.suffix.lower()
    meta = {
        "filename": p.name,
        "path": str(p.resolve()),
        "ext": ext,
        "size": p.stat().st_size,
    }
    if ext in SUPPORTED_TEXT_EXTS:
        text = load_text_file(p)
    elif ext == ".pdf":
        text = load_pdf_file(p)
    else:
        # Try reading as text fallback
        try:
            text = load_text_file(p)
        except Exception as e:
            raise ValueError(f"Unsupported document type {ext}: {e}")
    return text, meta

def load_documents_from_dir(dir_path: str, recursive: bool = True) -> List[Tuple[str, Dict]]:
    p = Path(dir_path)
    docs = []
    iterator = p.rglob("*") if recursive else p.iterdir()
    for f in iterator:
        if not f.is_file():
            continue
        if f.suffix.lower() in SUPPORTED_TEXT_EXTS or f.suffix.lower() == ".pdf":
            try:
                t, m = load_document(str(f))
                docs.append((t, m))
            except Exception as e:
                print(f"[loader] skip {f}: {e}")
    return docs
