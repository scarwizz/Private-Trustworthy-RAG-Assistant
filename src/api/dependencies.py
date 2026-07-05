# src/api/dependencies.py
"""
Dependency injection providers for the FastAPI application.
Manages singleton instances of the RAG pipeline and other services.
"""
import logging
import os
from functools import lru_cache
from typing import Optional, List, Dict, Any

from src.core.pipeline_explainable import RAGPipeline as ExplainableRAGPipeline
from src.models.llm_gguf import LocalGGUFLLM
from src.utils.config import Config
from src.data.loader import load_document  # Single-file loader
from src.data.chunker import chunk_text

logger = logging.getLogger("api.dependencies")


class RAGService:
    """
    Wrapper around the RAG pipeline to provide a clean service interface
    for dependency injection. Handles initialization and provides methods
    for ingestion and querying.
    """

    def __init__(self):
        """Initialize the RAG service."""
        self.config = Config()
        self._pipeline: Optional[ExplainableRAGPipeline] = None
        self._initialize_pipeline()

    def _initialize_pipeline(self) -> None:
        """Initialize the RAG pipeline singleton."""
        try:
            logger.info("Initializing RAG pipeline...")

            # Initialize GGUF model for generation
            model_path = self.config.gguf_model_path
            # Handle both absolute and relative paths
            if not model_path.startswith("/") and ":" not in model_path:
                # Relative path - make it relative to project root
                model_path = f"d:/TAI/{model_path}"

            generator = LocalGGUFLLM(
                model_path=model_path,
                n_ctx=self.config.gguf_n_ctx,
                n_batch=self.config.gguf_n_batch,
                n_threads=self.config.gguf_n_threads,
                n_gpu_layers=self.config.gguf_n_gpu_layers,
                temperature=self.config.gguf_temperature,  # Low temp for factual responses
                top_p=self.config.gguf_top_p,
                top_k=self.config.gguf_top_k,
                repeat_penalty=self.config.gguf_repeat_penalty,
                verbose=self.config.models.get("GGUF_VERBOSE", False),
            )

            self._pipeline = ExplainableRAGPipeline(
                generator=generator
            )
            logger.info("RAG pipeline initialized successfully with GGUF model")
        except Exception as e:
            logger.error(f"Failed to initialize RAG pipeline: {e}", exc_info=True)
            # Fallback to None generator (extractive only) if GGUF fails
            logger.warning("Falling back to extractive QA (no generation)")
            self._pipeline = ExplainableRAGPipeline()

    @property
    def pipeline(self) -> ExplainableRAGPipeline:
        """Get the initialized pipeline instance."""
        if self._pipeline is None:
            self._initialize_pipeline()
        return self._pipeline

    async def add_documents(self, file_paths: List[str]) -> int:
        """
        Add documents to the knowledge base.

        Args:
            file_paths: List of absolute paths to files to ingest

        Returns:
            Number of document chunks added
        """
        try:
            logger.info(f"Ingesting {len(file_paths)} file(s)")
            total_chunks = 0
            all_chunks = []

            for file_path in file_paths:
                try:
                    # Load document using the single-file loader
                    text, meta = load_document(file_path)
                    if not text.strip():
                        logger.warning(f"No text extracted from {file_path}")
                        continue

                    # Chunk the text with reduced size to avoid token overflow
                    chunks = chunk_text(
                        text,
                        max_chunk_chars=200,      # Reduced from 1000 to stay under token limit
                        overlap_chars=20,         # Reduced from 200
                        min_chunk_chars=50        # Reduced from 200
                    )

                    # Prepare chunks for embedding
                    chunk_texts = [chunk["text"] for chunk in chunks]
                    if not chunk_texts:
                        continue

                    # Generate embeddings using the pipeline's embedder
                    embeddings = self.pipeline.embedder.embed_texts(chunk_texts)

                    # Prepare records for upsert
                    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                        chunk_record = {
                            "id": f"{file_path}_{i}_{hash(chunk['text']) & 0xFFFFFFFF}",
                            "text": chunk["text"],
                            "embedding": embedding,
                            "metadata": {
                                **meta,
                                "chunk_index": i,
                                "start_char": chunk["start_char"],
                                "end_char": chunk["end_char"],
                                "source_file": file_path
                            }
                        }
                        all_chunks.append(chunk_record)

                    logger.info(f"Processed {file_path}: 1 document -> {len(chunks)} chunk(s)")

                except Exception as e:
                    logger.error(f"Failed to process file {file_path}: {e}", exc_info=True)
                    continue

            # Upsert all chunks to the vector store
            if all_chunks:
                self.pipeline.store.upsert_chunks(all_chunks)
                total_chunks = len(all_chunks)
                logger.info(f"Successfully ingested {total_chunks} chunks")
            else:
                # If we have file paths but no chunks, raise an error
                if file_paths:
                    raise ValueError("No text could be extracted from the provided files.")
                logger.warning("No chunks to ingest (no files provided)")

            return total_chunks

        except Exception as e:
            logger.error(f"Error during document ingestion: {e}", exc_info=True)
            raise

    async def answer_query(self, query: str, top_k: int = 3) -> Dict[str, Any]:
        """
        Process a query and return answer with sources.

        Args:
            query: User's question
            top_k: Number of results to retrieve

        Returns:
            Dictionary containing answer, sources, and debugging info
        """
        try:
            logger.info(f"Processing query: {query[:100]}...")
            result = self.pipeline.answer_query(
                query=query,
                top_k=top_k,
                return_sources=True,
                debug=False
            )

            # Ensure the result has the expected structure
            if not isinstance(result, dict):
                raise ValueError("Pipeline returned unexpected result type")

            # Normalize sources format
            sources = []
            for src in result.get("sources", []):
                sources.append({
                    "id": str(src.get("doc", "unknown")),
                    "content": str(src.get("snippet", ""))[:500],  # Ensure length limit
                    "metadata": src.get("raw_metadata", {}),
                    "score": float(src.get("score", 0.0))
                })

            return {
                "answer": str(result.get("answer", "")),
                "sources": sources,
                "query_id": f"qry_{hash(query) & 0xFFFFF}",
                "processing_time_ms": int(result.get("debug", {}).get("latency_ms", 0))
            }

        except Exception as e:
            logger.error(f"Error processing query: {e}", exc_info=True)
            raise


# Dependency providers using LRU cache for singleton behavior
@lru_cache()
def get_rag_service() -> RAGService:
    """
    Dependency provider that returns a singleton RAGService instance.
    Cached by LRU cache ensures only one instance is created.
    """
    return RAGService()


# Alternative dependency for direct pipeline access (if needed)
@lru_cache()
def get_rag_pipeline() -> ExplainableRAGPipeline:
    """
    Dependency provider that returns the RAG pipeline directly.
    Useful for endpoints that need direct pipeline access.
    """
    return get_rag_service().pipeline