import os
from pathlib import Path

base_cache = Path.home() / ".cache" / "huggingface" / "hub"
search_term = "embeddinggemma-300m"

found_paths = []
for root, dirs, files in os.walk(base_cache):
    if search_term.lower() in root.lower():
        found_paths.append(root)

if found_paths:
    print("✅ Found cached model folders:\n")
    for p in found_paths:
        print("-", p)
else:
    print("❌ No cached model found in Hugging Face cache.")
