#training/gen_rag_dataset.py
import random, json, os
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from chromadb import PersistentClient

# === Config ===
CHROMA_PATH = "./chroma_db"
OUTPUT_PATH = "./data/rag_synthetic.jsonl"
MODEL_NAME = "google/flan-t5-base"   # small enough
N_QA_PER_DOC = 3

# === Load ===
client = PersistentClient(path=CHROMA_PATH)
collection = client.get_collection("tai_documents")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
generator = pipeline("text2text-generation", model=model, tokenizer=tokenizer)

docs = [d for d in collection.get()["documents"]]

dataset = []

for doc in random.sample(docs, min(len(docs), 100)):  # limit for speed
    # Step 1: Generate a few synthetic questions about the doc
    q_prompt = f"Generate {N_QA_PER_DOC} factual questions that could be answered from this passage:\n\n{doc}"
    questions = generator(q_prompt, max_new_tokens=128, num_return_sequences=1)[0]['generated_text']
    questions = [q.strip() for q in questions.split('\n') if q.strip()]

    # Step 2: For each question, generate an answer grounded in the same doc
    for q in questions:
        a_prompt = f"Answer the following question based ONLY on this context.\n\nContext:\n{doc}\n\nQuestion: {q}\nAnswer:"
        answer = generator(a_prompt, max_new_tokens=256, num_return_sequences=1)[0]['generated_text']

        dataset.append({
            "context": doc,
            "question": q,
            "answer": answer
        })

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    for row in dataset:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")

print(f"✅ Synthetic dataset created at {OUTPUT_PATH} with {len(dataset)} examples")
