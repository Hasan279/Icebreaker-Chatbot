"""Ollama LLM and HuggingFace Inference API embedding interfaces."""

import logging
from typing import Any, List

import numpy as np
from huggingface_hub import InferenceClient
from llama_index.core.embeddings import BaseEmbedding
from llama_index.llms.ollama import Ollama
from pydantic import PrivateAttr

import config

logger = logging.getLogger(__name__)


class HFInferenceEmbedding(BaseEmbedding):
    """Synchronous wrapper around the HuggingFace Inference API.

    Uses InferenceClient directly to avoid asyncio event-loop conflicts
    with Gradio.
    """

    _client: InferenceClient = PrivateAttr()

    def __init__(self, model_name: str, token: str, **kwargs: Any) -> None:
        super().__init__(model_name=model_name, **kwargs)
        self._client = InferenceClient(model=model_name, token=token)

    def _embed(self, text: str) -> List[float]:
        result = self._client.feature_extraction(text)
        if isinstance(result, np.ndarray):
            return result.flatten().tolist()
        return list(result)

    def _get_text_embedding(self, text: str) -> List[float]:
        return self._embed(text)

    def _get_query_embedding(self, query: str) -> List[float]:
        return self._embed(query)

    async def _aget_text_embedding(self, text: str) -> List[float]:
        return self._embed(text)

    async def _aget_query_embedding(self, query: str) -> List[float]:
        return self._embed(query)


def create_huggingface_embedding() -> HFInferenceEmbedding:
    """Return an HFInferenceEmbedding using the configured model and token."""
    logger.info("Creating HuggingFace Inference API embedding: %s", config.EMBEDDING_MODEL_ID)
    return HFInferenceEmbedding(
        model_name=config.EMBEDDING_MODEL_ID,
        token=config.HUGGINGFACE_API_TOKEN,
    )


def create_ollama_llm(
    temperature: float = 0.0,
    max_new_tokens: int = 500,
) -> Ollama:
    """Return an Ollama LLM instance with the configured model and context window."""
    logger.info("Creating Ollama LLM: %s", config.LLM_MODEL_ID)
    return Ollama(
        model=config.LLM_MODEL_ID,
        base_url=config.OLLAMA_BASE_URL,
        temperature=temperature,
        request_timeout=120.0,
        context_window=config.CONTEXT_WINDOW,
        additional_kwargs={"num_ctx": config.CONTEXT_WINDOW},
    )


def change_llm_model(new_model_id: str) -> None:
    """Update the global LLM model ID at runtime."""
    config.LLM_MODEL_ID = new_model_id
    logger.info("LLM model changed to: %s", new_model_id)


create_ollama_embedding = create_huggingface_embedding
create_watsonx_embedding = create_huggingface_embedding
create_watsonx_llm = create_ollama_llm