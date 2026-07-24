"""Module for interfacing with local Ollama LLMs and HuggingFace embeddings."""

import logging
from typing import Optional

from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

import config

logger = logging.getLogger(__name__)


def create_huggingface_embedding() -> HuggingFaceEmbedding:
    """Creates a HuggingFace Embedding model for vector representation.

    Runs fully locally using sentence-transformers. The model is downloaded
    on first use and cached automatically.

    Returns:
        HuggingFaceEmbedding model using the configured model ID.
    """
    logger.info("Creating HuggingFace embedding model: %s", config.EMBEDDING_MODEL_ID)
    return HuggingFaceEmbedding(model_name=config.EMBEDDING_MODEL_ID)


def create_ollama_llm(
    temperature: float = 0.0,
    max_new_tokens: int = 500,
) -> Ollama:
    """Creates an Ollama LLM for generating responses.

    Args:
        temperature: Temperature for controlling randomness (0.0 to 1.0).
        max_new_tokens: Maximum number of new tokens to generate.

    Returns:
        Ollama LLM instance.
    """
    logger.info("Creating Ollama LLM: %s", config.LLM_MODEL_ID)
    return Ollama(
        model=config.LLM_MODEL_ID,
        base_url=config.OLLAMA_BASE_URL,
        temperature=temperature,
        request_timeout=120.0,
    )


def change_llm_model(new_model_id: str) -> None:
    """Change the LLM model used globally.

    Args:
        new_model_id: New Ollama model tag (e.g. 'phi3.5', 'llama3').
    """
    config.LLM_MODEL_ID = new_model_id
    logger.info("LLM model changed to: %s", new_model_id)


# ---------------------------------------------------------------------------
# Backwards-compatible aliases
# ---------------------------------------------------------------------------
create_ollama_embedding = create_huggingface_embedding
create_watsonx_embedding = create_huggingface_embedding
create_watsonx_llm = create_ollama_llm