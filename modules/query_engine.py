"""Module for querying indexed LinkedIn profile data."""

import logging
from typing import Any, Optional

from llama_index.core import VectorStoreIndex, PromptTemplate

from modules.llm_interface import create_ollama_llm
import config

logger = logging.getLogger(__name__)


def generate_initial_facts(index: VectorStoreIndex) -> str:
    """Generates interesting facts about the person's career or education.

    Args:
        index: VectorStoreIndex containing the LinkedIn profile data.

    Returns:
        String containing interesting facts about the person.
    """
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
    """Answers the user's question using the vector database and the LLM.

    Args:
        index: VectorStoreIndex containing the LinkedIn profile data.
        user_query: The user's question.

    Returns:
        Response object containing the answer to the user's question.
    """
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