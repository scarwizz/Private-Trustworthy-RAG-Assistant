#src\utils\config.py

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
    "EMBEDDING_MODEL": "D:/TAI/models/bge-base-en-v1.5",

    # SLM (Generation) - ~250M parameters, instruction-tuned for Q&A
    "LLM_MODEL": "google/flan-t5-base",

    # GGUF Model Configuration
    "GGUF_MODEL_PATH": "models/Phi-3-mini-4k-instruct-q4_k_m.gguf",
    "GGUF_N_CTX": 4096,                     # Context window size
    "GGUF_N_BATCH": 8,                      # Batch size for prompt processing
    "GGUF_N_THREADS": None,                 # Number of CPU threads to use (None = auto)
    "GGUF_N_GPU_LAYERS": 0,                 # Number of layers to offload to GPU (0 = CPU only)
    "GGUF_TEMPERATURE": 0.1,                # Low temperature for factual responses
    "GGUF_TOP_P": 0.95,                     # Nucleus sampling threshold
    "GGUF_TOP_K": 40,                       # Top-k sampling limit
    "GGUF_REPEAT_PENALTY": 1.1,             # Penalty for repeating tokens
    "LLM_MODEL_PATH": "./models",             # Path to local GGUF models
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
    # "VECTOR_DB_DIR": str(BASE_DIR / "src" / "db" / "chromadb_data"),
    "VECTOR_DB_DIR": "./chroma_db",

    
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
    "CHUNK_SIZE": 256,
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
class Config:
    """Wrapper class for CONFIG dict (for backward compatibility)."""
    def __init__(self):
        self.models = CONFIG["MODELS"]
        self.paths = CONFIG["PATHS"]
        self.system = CONFIG["SYSTEM"]

    @property
    def gemma_model_path(self):
        return self.models["EMBEDDING_MODEL"]

    @property
    def llm_model_path(self):
        return self.models["LLM_MODEL"]

    # GGUF Model Properties
    @property
    def gguf_model_path(self):
        return self.models["GGUF_MODEL_PATH"]

    @property
    def gguf_n_ctx(self):
        return self.models["GGUF_N_CTX"]

    @property
    def gguf_n_batch(self):
        return self.models["GGUF_N_BATCH"]

    @property
    def gguf_n_gpu_layers(self):
        return self.models["GGUF_N_GPU_LAYERS"]

    @property
    def gguf_n_threads(self):
        return self.models["GGUF_N_THREADS"]

    @property
    def gguf_temperature(self):
        return self.models["GGUF_TEMPERATURE"]

    @property
    def gguf_top_p(self):
        return self.models["GGUF_TOP_P"]

    @property
    def gguf_top_k(self):
        return self.models["GGUF_TOP_K"]

    @property
    def gguf_repeat_penalty(self):
        return self.models["GGUF_REPEAT_PENALTY"]