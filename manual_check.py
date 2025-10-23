# manual_check.py

import sys
import os
import torch
import numpy as np

# Temporarily add 'src' directory to Python path to import config
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

try:
    from utils.config import CONFIG
    from sentence_transformers import SentenceTransformer
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
except ImportError as e:
    print(f"FATAL ERROR: Core library not found. Have you run 'uv sync'? Error: {e}")
    sys.exit(1)

# --- Define Model IDs and Cache Path ---
EMBEDDING_MODEL_ID = CONFIG["MODELS"]["EMBEDDING_MODEL"]
SLM_MODEL_ID = CONFIG["MODELS"]["LLM_MODEL"]
CACHE_DIR = os.path.abspath(CONFIG["PATHS"]["MODEL_CACHE_DIR"]) # Uses the D:\TAI\local_models_cache directory

# Set a dummy sentence for testing
TEST_SENTENCE = "This is a quick test to ensure the models load from the local cache."

def check_embedding_model():
    print("\n--- 🔎 1. Checking EMBEDDING Model (google/embeddinggemma-300M) ---")
    
    # Check if files are in the expected path (not strictly necessary but useful for confirmation)
    gemma_path_in_cache = os.path.join(CACHE_DIR, f'models--google--{EMBEDDING_MODEL_ID.replace("/", "--")}')
    
    if not os.path.exists(gemma_path_in_cache) and not os.path.exists(os.path.expanduser('~/.cache/huggingface/hub')):
        print("❌ Model directory not found in project cache or default C: drive cache.")
        return False
        
    try:
        # Load the model. It automatically finds the cached location.
        # Set local_files_only=True to force offline loading (proving the models were downloaded)
        model = SentenceTransformer(EMBEDDING_MODEL_ID, local_files_only=True)
        print(f"✅ Model found and loaded. Dimension: {model.get_sentence_embedding_dimension()}")
        
        # Test embedding generation
        embeddings = model.encode([TEST_SENTENCE])
        print(f"✅ Embedding successful. Output shape: {embeddings.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ ERROR: Failed to load Embedding Model: {e}")
        print("ACTION: Ensure the license is accepted on Hugging Face.")
        return False

def check_slm_model():
    print("\n--- 💬 2. Checking SLM (google/flan-t5-base) ---")
    
    try:
        # Load the tokenizer and model. Use local_files_only=True.
        tokenizer = AutoTokenizer.from_pretrained(SLM_MODEL_ID, local_files_only=True)
        model = AutoModelForSeq2SeqLM.from_pretrained(SLM_MODEL_ID, local_files_only=True, device_map="cpu")
        
        print(f"✅ Tokenizer and Model loaded successfully.")
        
        # Test a simple inference (generation) flow
        input_ids = tokenizer("The capital of France is", return_tensors="pt")
        outputs = model.generate(**input_ids, max_new_tokens=10)
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        print(f"✅ Simple generation successful. Output: '{generated_text}'")
        
        return True
        
    except Exception as e:
        print(f"❌ ERROR: Failed to load SLM: {e}")
        print("ACTION: Ensure 'accelerate' is installed. This model is ~1GB and needs sufficient RAM.")
        return False


if __name__ == "__main__":
    
    print("=== STARTING MANUAL SANITY CHECK ===")
    emb_ok = check_embedding_model()
    slm_ok = check_slm_model()
    
    print("\n" + "="*50)
    if emb_ok and slm_ok:
        print("✨ FINAL CONFIRMATION: ALL CORE MODELS ARE READY AND OFFLINE. ✨")
        print("You can now safely proceed with coding the pipeline logic.")
    else:
        print("🛑 WARNING: One or more core models failed to load. Cannot proceed offline.")
    print("="*50)