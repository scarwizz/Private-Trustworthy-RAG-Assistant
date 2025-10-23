import os

# Base project directory
BASE_DIR = r"D:\TAI"

# Folder structure (relative to BASE_DIR)
structure = {
    "src": [
        "ui/components",
        "ui/static",
        "core",
        "models",
        "data/raw",
        "data/processed",
        "data/embeddings",
        "db",
        "security",
        "utils",
        "tests"
    ],
    "notebooks": [],
    "docs": [],
    "assets": []
}

# Files to create at base level (if not present)
base_files = [
    "main.py",
    "README.md",
    "pyproject.toml",
    ".gitignore"
]

# Template contents for important files
templates = {
    "__init__.py": "# Package initializer\n",
    "src/utils/config.py": '''"""
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
''',

    "src/utils/logger.py": '''import logging

def get_logger(name="RAGAssistant"):
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("[%(levelname)s] %(asctime)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger
''',

    "src/core/pipeline.py": '''"""
Main RAG Pipeline: handles retrieval → generation → response formatting.
"""
from src.utils.logger import get_logger
logger = get_logger()

class RAGPipeline:
    def __init__(self, retriever, generator):
        self.retriever = retriever
        self.generator = generator

    def query(self, text):
        logger.info(f"Received query: {text}")
        retrieved_docs = self.retriever.retrieve(text)
        answer = self.generator.generate(text, retrieved_docs)
        return {
            "answer": answer,
            "sources": retrieved_docs
        }
''',

    "src/ui/app.py": '''"""
Local UI application (Streamlit or Flask interface)
"""
import streamlit as st
from src.core.pipeline import RAGPipeline

def main():
    st.title("🔒 Private RAG Assistant")
    query = st.text_input("Ask a question:")
    if st.button("Submit") and query:
        st.write("Processing query locally...")
        # Placeholder response
        st.success("Answer will appear here once pipeline is connected.")

if __name__ == "__main__":
    main()
''',
}

# Function to create folders and files
def make_structure():
    for base, subfolders in structure.items():
        for sub in subfolders:
            path = os.path.join(BASE_DIR, base, sub)
            os.makedirs(path, exist_ok=True)
            # Add __init__.py in every directory
            init_path = os.path.join(path, "__init__.py")
            if not os.path.exists(init_path):
                with open(init_path, "w", encoding="utf-8") as f:
                    f.write("# Package initializer\n")

    # Create base files if missing
    for file in base_files:
        path = os.path.join(BASE_DIR, file)
        if not os.path.exists(path):
            open(path, "w", encoding="utf-8").close()

    # Add template contents
    for rel_path, content in templates.items():
        abs_path = os.path.join(BASE_DIR, rel_path)
        os.makedirs(os.path.dirname(abs_path), exist_ok=True)
        with open(abs_path, "w", encoding="utf-8") as f:
            f.write(content)

    print("✅ Project structure created successfully at:", BASE_DIR)


if __name__ == "__main__":
    make_structure()
