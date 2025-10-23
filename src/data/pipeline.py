# src/data/pipeline.py
"""
High-level ingestion pipeline:
- load document(s)
- chunk
- embed via GemmaEmbeddings
- upsert into EmbedStore (Chroma or FAISS)
"""
import uuid
from typing import List
from pathlib import Path
from src.data.loader import load_document, load_documents_from_dir
from src.data.chunker import chunk_text
from src.models.embeddings import GemmaEmbeddings
from src.data.embed_store import EmbedStore
from src.security.encryptor import maybe_encrypt_metadata, Encryptor
from src.utils.logger import get_logger

logger = get_logger("data_pipeline")

DEFAULT_CHUNK_SIZE = 1000
DEFAULT_OVERLAP = 200

def _make_chunk_record(doc_meta: dict, chunk: dict, embedding: List[float], encryptor: Encryptor = None):
    rec_id = str(uuid.uuid4())
    metadata = {
        "source_filename": doc_meta.get("filename"),
        "source_path": doc_meta.get("path"),
        "source_ext": doc_meta.get("ext"),
        "start_char": chunk.get("start_char"),
        "end_char": chunk.get("end_char")
    }
    if encryptor is not None:
        metadata = maybe_encrypt_metadata(metadata, encryptor)
    return {
        "id": rec_id,
        "embedding": embedding,
        "metadata": metadata,
        "text": chunk["text"]
    }

def ingest_document(
    path: str,
    embedder: GemmaEmbeddings,
    store: EmbedStore,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_OVERLAP,
    encryptor: Encryptor = None,
):
    text, doc_meta = load_document(path)
    logger.info(f"Ingesting {doc_meta['filename']} (size={doc_meta['size']})")
    chunks = chunk_text(text, max_chunk_chars=chunk_size, overlap_chars=chunk_overlap)
    logger.info(f"{len(chunks)} chunks created from {doc_meta['filename']}")

    # embed in batches using embedder.embed_texts
    texts = [c["text"] for c in chunks]
    embeddings = embedder.embed_texts(texts)
    records = []
    for c, emb in zip(chunks, embeddings):
        records.append(_make_chunk_record(doc_meta, c, emb, encryptor))
    store.upsert_chunks(records)
    logger.info(f"Upserted {len(records)} chunks for {doc_meta['filename']}")

def ingest_directory(
    dir_path: str,
    embedder: GemmaEmbeddings,
    store: EmbedStore,
    recursive: bool = True,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_OVERLAP,
    encryptor: Encryptor = None,
):
    docs = load_documents_from_dir(dir_path, recursive=recursive)
    logger.info(f"Found {len(docs)} documents in {dir_path}")
    for text, meta in docs:
        # chunk
        chunks = chunk_text(text, max_chunk_chars=chunk_size, overlap_chars=chunk_overlap)
        texts = [c["text"] for c in chunks]
        embeddings = embedder.embed_texts(texts)
        records = []
        for c, emb in zip(chunks, embeddings):
            records.append({
                "id": str(uuid.uuid4()),
                "embedding": emb,
                "metadata": {
                    "source_filename": meta.get("filename"),
                    "source_path": meta.get("path"),
                    "source_ext": meta.get("ext"),
                    "start_char": c.get("start_char"),
                    "end_char": c.get("end_char")
                },
                "text": c["text"]
            })
        store.upsert_chunks(records)
        logger.info(f"Ingested {meta.get('filename')} -> {len(records)} chunks")

# CLI for quick use
if __name__ == "__main__":
    import argparse
    from src.utils.config import Config

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", required=True, help="file or directory to ingest")
    parser.add_argument("--use_chroma", action="store_true", default=False, help="Prefer Chroma if installed")
    parser.add_argument("--chroma_dir", default="./chroma_db")
    parser.add_argument("--faiss_dir", default="./faiss_index")
    parser.add_argument("--chunk_size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--overlap", type=int, default=DEFAULT_OVERLAP)
    parser.add_argument("--encrypt", action="store_true", default=False)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    cfg = Config()
    # instantiate embedder
    embedder = GemmaEmbeddings(model_name=cfg.gemma_model_path, device=None, local_only=True, batch_size=args.batch_size)
    store = EmbedStore(use_chroma=args.use_chroma, chroma_dir=args.chroma_dir, faiss_dir=args.faiss_dir, embedding_dim=embedder.dim or 1536)
    encryptor = Encryptor() if args.encrypt else None

    inp = Path(args.input)
    if inp.is_dir():
        ingest_directory(str(inp), embedder, store, recursive=True, chunk_size=args.chunk_size, chunk_overlap=args.overlap, encryptor=encryptor)
    else:
        ingest_document(str(inp), embedder, store, chunk_size=args.chunk_size, chunk_overlap=args.overlap, encryptor=encryptor)
