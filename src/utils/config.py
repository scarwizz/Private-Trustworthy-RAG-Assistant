"""
Centralized configuration file for Private RAG Assistant.
Defines model IDs, disk paths, and core system settings.
"""
import os
from pathlib import Path
import torch

# --- BASE DIRECTORY SETUP ---
# Dynamically determines the project root path (D:\TAI)
# BASE_DIR is two levels up from this script (src/utils/config.py -> D:\TAI)
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# --- MODEL IDENTIFIERS (Hugging Face Hub IDs) ---
# NOTE: These are the exact IDs used by the 'transformers' and 'sentence-transformers' libraries.
MODEL_IDS = {
    # Embedding Model (Retrieval) - ~300M parameters, efficient for RAG index creation
    "EMBEDDING_MODEL": "google/embeddinggemma-300M",
    
    # SLM (Generation) - ~250M parameters, instruction-tuned for Q&A
    "LLM_MODEL": "google/flan-t5-base"
}

# --- PATHS & STORAGE ---
# Defines persistent disk locations relative to the project root (BASE_DIR)
PATHS = {
    # Directory where model weights are downloaded and cached.
    "MODEL_CACHE_DIR": str(BASE_DIR / "local_models_cache"), 
    
    # User-provided raw documents (PDFs, .txt, etc.)
    "DATA_RAW": str(BASE_DIR / "src" / "data" / "raw"),
    
    # Processed text chunks before vectorization
    "DATA_PROCESSED": str(BASE_DIR / "src" / "data" / "processed"),
    
    # Location of the ChromaDB/FAISS index files
    "VECTOR_DB_DIR": str(BASE_DIR / "src" / "db" / "chromadb_data"),
    
    # Local file for storing chat history and metadata (e.g., SQLite file)
    "CHAT_HISTORY_DB": str(BASE_DIR / "src" / "db" / "chat_history.sqlite")
}

# --- CORE SYSTEM SETTINGS ---
SYSTEM_CONFIG = {
    # Boolean: True attempts to use CUDA/GPU; False uses CPU (default for local privacy focus)
    "USE_GPU": True if torch.cuda.is_available() else False,
    
    # Maximum size of the context window (retrieved chunks + query) passed to the LLM
    "MAX_CONTEXT_TOKENS": 1024,
    
    # Document processing settings
    "CHUNK_SIZE": 512,
    "CHUNK_OVERLAP": 50,
    
    # Number of top documents/chunks to retrieve from the vector database for RAG
    "TOP_K_RETRIEVAL": 3,
}

# Combine all configurations into a single accessible dictionary
CONFIG = {
    "MODELS": MODEL_IDS,
    "PATHS": PATHS,
    "SYSTEM": SYSTEM_CONFIG
}

# --- Ensure Directories Exist on First Run ---
# This is a good practice for project stability.
def setup_directories():
    """Creates all necessary local storage folders if they do not exist."""
    for key, path in PATHS.items():
        # Exclude the DB file path, as it's a file, not a directory
        if key not in ["CHAT_HISTORY_DB"]:
            os.makedirs(path, exist_ok=True)
            # print(f"Ensured directory exists: {path}")

# Run setup when the script is imported/executed
setup_directories()