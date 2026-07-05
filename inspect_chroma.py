import chromadb

client = chromadb.PersistentClient(path="./chroma_db")
col = client.get_collection("cv_docs")  # or tai_documents
print("Documents in collection:", col.count())

peek = col.peek()
for i, doc in enumerate(peek["documents"]):
    print(f"\n--- Document {i} ---")
    print(doc[:300])
