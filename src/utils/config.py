"""
Centralized configuration file for Private RAG Assistant.
Edit paths and model settings here.
"""
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CONFIG = {
    "models": {
        "embedding_model": "gemma",
        "llm_model": "flan-t5-small"
    },
    "paths": {
        "data_raw": os.path.join(BASE_DIR, "data", "raw"),
        "data_processed": os.path.join(BASE_DIR, "data", "processed"),
        "embeddings": os.path.join(BASE_DIR, "data", "embeddings"),
        "vector_db": os.path.join(BASE_DIR, "db"),
    },
    "system": {
        "use_gpu": True,
        "max_context_tokens": 1024
    }
}
