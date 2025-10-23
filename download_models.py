# download_models.py

import os
import sys
import torch
from huggingface_hub import login
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
import subprocess

# --- 1. Import Configuration ---
# Assuming D:\TAI is the root, we import config from src/utils/
try:
    # We use uv run to execute this, so we add the src dir to path temporarily
    # to import the config file cleanly.
    sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))
    from utils.config import CONFIG # Import the CONFIG dictionary
    
except ImportError:
    print("❌ ERROR: Could not import CONFIG from src/utils/config.py.")
    print("Ensure the directory structure and config.py file are correct.")
    sys.exit(1)

# --- 2. Configuration Variables ---
# Note: We use the *Hugging Face model IDs* for download, as they are the source.
EMBEDDING_MODEL_ID = "google/embeddinggemma-300M"
SLM_MODEL_ID = "google/flan-t5-base"

# The cache location will be a subdirectory of D:\TAI
MODEL_CACHE_DIR = "./local_models_cache"

# --- 3. Setup Logic ---
def setup_models():
    """Authenticates, creates cache folder, and downloads/caches models locally."""
    print("--- 🔑 STEP 1: Authenticating with Hugging Face ---")
    print("NOTE: Gemma models require accepting Google's license terms first.")
    
    # Hugging Face Login (Prompts for token)
    try:
        login()
        print("✅ Successfully logged in to Hugging Face Hub.")
    except Exception as e:
        print(f"\n❌ ERROR: Hugging Face login failed. Details: {e}")
        print("Please check your token and accept the model licenses.")
        sys.exit(1)

    # Setup Cache Directory and Device
    print(f"\n--- ⬇️ STEP 2: Downloading and Caching Models to {MODEL_CACHE_DIR} ---")
    os.makedirs(MODEL_CACHE_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() and CONFIG['system']['use_gpu'] else "cpu"
    print(f"Device for model loading set to: {device.upper()}.")
    
    # Set the cache directory for Hugging Face to ensure local storage
    os.environ['HF_HOME'] = os.path.abspath(MODEL_CACHE_DIR)

    # --- 3a. Download SLM (Flan-T5-base) ---
    print(f"\nDownloading SLM (Generation Model): {SLM_MODEL_ID}...")
    try:
        # T5-Base is ~250M params. Low_cpu_mem_usage helps on CPU.
        AutoTokenizer.from_pretrained(SLM_MODEL_ID, cache_dir=os.environ['HF_HOME'])
        AutoModelForSeq2SeqLM.from_pretrained(
            SLM_MODEL_ID, 
            cache_dir=os.environ['HF_HOME'], 
            device_map="cpu", # Force CPU mapping for local setup simplicity
            low_cpu_mem_usage=True 
        )
        print(f"✅ SLM ({SLM_MODEL_ID}) successfully cached.")
    except Exception as e:
        print(f"❌ ERROR downloading SLM: {e}")
        print("Consider a smaller model like flan-t5-small if you face memory issues.")
        sys.exit(1)

    # --- 3b. Download Embedding Model (Gemma-2-embedding) ---
    print(f"\nDownloading Embedding Model (Retrieval Model): {EMBEDDING_MODEL_ID}...")
    try:
        # SentenceTransformer handles the download and uses the HF_HOME cache
        SentenceTransformer(EMBEDDING_MODEL_ID, device=device)
        print(f"✅ Embedding Model ({EMBEDDING_MODEL_ID}) successfully cached.")
    except Exception as e:
        print(f"❌ ERROR downloading Embedding Model: {e}")
        print("Ensure you have accepted the Gemma 2 license on Hugging Face.")
        sys.exit(1)
        
    print("\n" + "="*50)
    print("✨ ALL MODELS CACHED LOCALLY. SETUP COMPLETE! ✨")
    print(f"Models are in: {os.path.abspath(MODEL_CACHE_DIR)}")
    print("="*50)

if __name__ == "__main__":
    setup_models()

