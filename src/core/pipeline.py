"""
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
