# src/core/generator.py
"""
Local Flan-T5 generator wrapper.

Implements:
    class FlanGenerator:
        generate(query: str, contexts: List[str], max_new_tokens: int = 128, template: str = "short_answer") -> str

Two prompt templates are provided:
    - short_answer: concise answer
    - detailed_answer: more verbose, step-by-step
"""
from typing import List, Optional
import logging
import math

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from src.utils.config import CONFIG

logger = logging.getLogger(__name__)


PROMPT_TEMPLATES = {
    "short_answer": (
        "You are a helpful assistant. Use the numbered context passages below to answer the question concisely.\n\n"
        "CONTEXT:\n{context}\n\n"
        "QUESTION: {query}\n\n"
        "Answer briefly and cite the context numbers used (e.g., [1], [2]).\n"
    ),
    "detailed_answer": (
        "You are an expert assistant. Use the context passages below to answer the question thoroughly. "
        "First give a short summary, then a more detailed explanation. Cite context numbers used.\n\n"
        "CONTEXT:\n{context}\n\n"
        "QUESTION: {query}\n"
        "Provide a clear, sourced answer.\n"
    ),
}


class FlanGenerator:
    """
    Wraps a local Flan-T5 model for deterministic generation.

    - Loads tokenizer + model lazily.
    - Uses deterministic decoding (do_sample=False).
    """

    def __init__(self, model_name: Optional[str] = None, device: Optional[str] = None):
        self.model_name = model_name or CONFIG["MODELS"].get("GENERATOR_MODEL", "google/flan-t5-base")
        # Determine device
        if device:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        self.model = None
        # Config constraints
        self.max_context_tokens = CONFIG["SYSTEM"].get("MAX_CONTEXT_TOKENS", 2048)
        self.default_max_new_tokens = CONFIG["SYSTEM"].get("MAX_GENERATION_TOKENS", 128)

    def _ensure_model_loaded(self):
        if self.model is not None and self.tokenizer is not None:
            return
        try:
            logger.info("Loading generator model '%s' to device %s", self.model_name, self.device)
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, local_files_only=True)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name, local_files_only=True)
            self.model.to(self.device)
        except Exception as e:
            logger.exception("Failed to load generator model.")
            raise RuntimeError(f"Failed to load generator model '{self.model_name}': {e}") from e

    def _build_context_block(self, contexts: List[str], metadata_list: Optional[List[dict]] = None) -> str:
        """
        Build a single string that enumerates contexts with source tags, respecting token budget.
        Uses tokenizer to count tokens and truncates lower-priority contexts if necessary.
        """
        if self.tokenizer is None:
            # load tokenizer only for token counting if needed
            self._ensure_model_loaded()

        # attach metadata (filename/id) when available
        numbered = []
        for idx, c in enumerate(contexts, start=1):
            tag = ""
            if metadata_list and idx - 1 < len(metadata_list):
                md = metadata_list[idx - 1] or {}
                fname = md.get("filename") or md.get("source") or md.get("file") or md.get("doc_id")
                if fname:
                    tag = f" (source: {fname})"
            numbered.append(f"[{idx}]{tag}: {c.strip()}")

        # Try to ensure the whole context fits in max_context_tokens by iteratively truncating tail contexts.
        joined = "\n\n".join(numbered)
        tokenized = self.tokenizer.encode(joined, truncation=False, add_special_tokens=False)
        token_len = len(tokenized)
        if token_len <= self.max_context_tokens:
            return joined

        # If it exceeds, we remove lowest-priority contexts (end of list) until it fits,
        # then further truncate the last remaining context if necessary.
        trimmed_numbered = numbered.copy()
        while len(trimmed_numbered) > 1:
            trimmed_numbered.pop()  # drop last
            candidate = "\n\n".join(trimmed_numbered)
            token_len = len(self.tokenizer.encode(candidate, truncation=False, add_special_tokens=False))
            if token_len <= self.max_context_tokens:
                return candidate

        # If only one context remains and still too long, truncate its tokens
        last = trimmed_numbered[0]
        # we'll keep max_context_tokens - 50 tokens for safety for other prompt pieces
        reserve = max(50, int(0.05 * self.max_context_tokens))
        allowed = max(10, self.max_context_tokens - reserve)
        # decode tokens up to allowed and then decode back to text
        enc = self.tokenizer.encode(last, truncation=True, add_special_tokens=False)[:allowed]
        truncated_text = self.tokenizer.decode(enc, clean_up_tokenization_spaces=True, skip_special_tokens=True)
        return truncated_text

    def generate(
        self,
        query: str,
        contexts: List[str],
        metadata_list: Optional[List[dict]] = None,
        max_new_tokens: Optional[int] = None,
        template: str = "short_answer",
    ) -> str:
        """
        Generate an answer given a query and list of context strings.

        :param query: user query
        :param contexts: list of textual contexts (ordered by relevance)
        :param metadata_list: optional list of metadata dicts corresponding to contexts
        :param max_new_tokens: max tokens to generate (defaults from config)
        :param template: 'short_answer' or 'detailed_answer'
        :returns: generated plain text
        """
        if not query:
            raise ValueError("Query must be provided.")
        if contexts is None:
            contexts = []

        self._ensure_model_loaded()
        max_new_tokens = int(max_new_tokens) if max_new_tokens is not None else self.default_max_new_tokens
        template = template if template in PROMPT_TEMPLATES else "short_answer"

        # Build context block respecting token budget
        context_block = ""
        if contexts:
            try:
                context_block = self._build_context_block(contexts, metadata_list=metadata_list)
            except Exception:
                # fallback: simple concatenation of first N contexts (defensive)
                logger.exception("Context trimming failed; using simple concatenation fallback.")
                context_block = "\n\n".join(contexts[: max(1, min(len(contexts), 5))])
        else:
            context_block = "No relevant context available."

        prompt = PROMPT_TEMPLATES[template].format(context=context_block, query=query)

        # Tokenize and generate deterministically
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=self.max_context_tokens).to(
            self.device
        )
        try:
            with torch.no_grad():
                out = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    num_return_sequences=1,
                    early_stopping=True,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
                decoded = self.tokenizer.decode(out[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
                return decoded.strip()
        except Exception as e:
            logger.exception("Generation failed.")
            raise RuntimeError(f"Generation failed: {e}") from e
