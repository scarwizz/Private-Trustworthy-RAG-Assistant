# src/query/transformer.py
"""
Query transformation and routing module for the Private RAG Assistant.
Implements HyDE (Hypothetical Document Embeddings) and query routing.
"""
from typing import Optional, Tuple
from src.models.llm_gguf import LocalGGUFLLM
from src.utils.config import Config
from src.utils.logger import get_logger

logger = get_logger("query.transformer")


class QueryTransformer:
    """
    Transforms and routes queries for improved retrieval.
    """

    def __init__(self, llm: Optional[LocalGGUFLLM] = None):
        """
        Initialize the query transformer.

        Args:
            llm: Language model to use for transformations.
                 If None, uses the default GGUF model from config.
        """
        self.config = Config()
        self.llm = llm or LocalGGUFLLM(
            model_path=self.config.models["GGUF_MODEL_PATH"],
            n_ctx=self.config.models["GGUF_N_CTX"],
            n_batch=self.config.models["GGUF_N_BATCH"],
            n_threads=self.config.models["GGUF_N_THREADS"],
            n_gpu_layers=self.config.models["GGUF_N_GPU_LAYERS"],
            temperature=self.config.models["GGUF_TEMPERATURE"],
            top_p=self.config.models["GGUF_TOP_P"],
            top_k=self.config.models["GGUF_TOP_K"],
            repeat_penalty=self.config.models["GGUF_REPEAT_PENALTY"],
            verbose=self.config.models.get("GGUF_VERBOSE", False)
        )

    def hyde_transform(self, query: str) -> str:
        """
        Generate a hypothetical document (HyDE) for the given query.
        This helps bridge the vocabulary gap between query and documents.

        Args:
            query: Original user query

        Returns:
            Hypothetical answer string that embeds well with the target documents
        """
        prompt = f"""Please write a concise, factual paragraph that directly answers the following question.
        Focus on providing accurate information that would be found in a reliable source.
        Do not add extra commentary or explanation.

        Question: {query}

        Answer:"""

        try:
            # Use low temperature for factual, consistent output
            hypothetical = self.llm.generate(
                prompt=prompt,
                max_new_tokens=150,
                temperature=0.1,
                top_p=0.9
            )
            # Clean up the response
            hypothetical = hypothetical.strip()
            if not hypothetical:
                # Fallback to original question if generation fails
                return query
            return hypothetical
        except Exception as e:
            logger.warning(f"HyDE generation failed: {e}. Falling back to original query.")
            return query

    def route_query(self, query: str) -> str:
        """
        Route query to broad categories for potential retrieval strategy adjustment.
        Currently distinguishes between definitional and procedural queries.

        Args:
            query: User query string

        Returns:
            One of: "definitional", "procedural", "general"
        """
        # Simple rule-based routing based on question starters
        query_lower = query.lower().strip()

        # Definitional patterns
        definitional_starts = [
            "what is", "what are", "who is", "who are", "define", "definition",
            "meaning of", "means", "stands for", "refers to", "is a", "are"
        ]

        # Procedural patterns
        procedural_starts = [
            "how to", "how do", "how does", "how can", "steps to", "way to",
            "method for", "process of", "procedure", "tutorial", "guide",
            "explain how", "show me how"
        ]

        # Check for definitional
        for start in definitional_starts:
            if query_lower.startswith(start):
                return "definitional"

        # Check for procedural
        for start in procedural_starts:
            if query_lower.startswith(start):
                return "procedural"

        # Default to general
        return "general"

    def transform(self, query: str) -> Tuple[str, str]:
        """
        Apply both HyDE transformation and routing to a query.

        Args:
            query: Original user query

        Returns:
            Tuple of (hyde_query, route_category)
        """
        hyde_query = self.hyde_transform(query)
        route = self.route_query(query)
        return hyde_query, route