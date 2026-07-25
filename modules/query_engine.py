"""Query engine for RAG-based Q&A over LinkedIn profile data."""

import logging
from typing import Any

from llama_index.core import VectorStoreIndex, PromptTemplate

from modules.llm_interface import create_ollama_llm
import config

logger = logging.getLogger(__name__)


def generate_initial_facts(index: VectorStoreIndex) -> str:
    """Generate three interesting facts about the profiled person."""
    logger.info("Generating initial facts from profile...")
    llm = create_ollama_llm(
        temperature=config.TEMPERATURE,
        max_new_tokens=config.MAX_NEW_TOKENS,
    )

    prompt_template = PromptTemplate(config.INITIAL_FACTS_TEMPLATE)

    query_engine = index.as_query_engine(
        llm=llm,
        similarity_top_k=config.SIMILARITY_TOP_K,
        text_qa_template=prompt_template,
    )

    response = query_engine.query(
        "List 3 interesting facts about this person's career or education."
    )

    logger.info("Initial facts generated.")
    return str(response)


def answer_user_query(index: VectorStoreIndex, user_query: str) -> Any:
    """Answer a free-form question using the indexed profile data."""
    logger.info("Answering user query: %s", user_query)
    llm = create_ollama_llm(
        temperature=config.TEMPERATURE,
        max_new_tokens=config.MAX_NEW_TOKENS,
    )

    prompt_template = PromptTemplate(config.USER_QUESTION_TEMPLATE)

    query_engine = index.as_query_engine(
        llm=llm,
        similarity_top_k=config.SIMILARITY_TOP_K,
        text_qa_template=prompt_template,
    )

    response = query_engine.query(user_query)
    logger.info("Query answered.")
    return response