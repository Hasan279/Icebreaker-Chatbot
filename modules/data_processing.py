"""Module for processing LinkedIn profile data."""

import json
import logging
from typing import Dict, List, Any, Optional

from llama_index.core import Document, VectorStoreIndex, Settings
from llama_index.core.node_parser import SentenceSplitter

from modules.llm_interface import create_ollama_embedding
import config

logger = logging.getLogger(__name__)


def split_profile_data(profile_data: Dict[str, Any]) -> List:
    """Splits the LinkedIn profile JSON data into nodes.

    Args:
        profile_data: LinkedIn profile data dictionary.

    Returns:
        List of document nodes.
    """
    logger.info("Splitting profile data into nodes...")
    profile_text = json.dumps(profile_data, indent=2)
    document = Document(text=profile_text)

    splitter = SentenceSplitter(chunk_size=config.CHUNK_SIZE)
    nodes = splitter.get_nodes_from_documents([document])

    logger.info("Split profile into %d nodes", len(nodes))
    return nodes


def create_vector_database(nodes: List) -> Optional[VectorStoreIndex]:
    """Stores the document chunks (nodes) in a vector database.

    Args:
        nodes: List of document nodes to be indexed.

    Returns:
        VectorStoreIndex or None if indexing fails.
    """
    try:
        logger.info("Creating vector database with %d nodes...", len(nodes))
        embed_model = create_ollama_embedding()

        # Set embedding model globally for this index build
        Settings.embed_model = embed_model

        index = VectorStoreIndex(nodes, embed_model=embed_model, show_progress=True)
        logger.info("Vector database created successfully.")
        return index
    except Exception as e:
        logger.error("Failed to create vector database: %s", e)
        return None


def verify_embeddings(index: VectorStoreIndex) -> bool:
    """Verify that all nodes have been properly embedded.

    Args:
        index: VectorStoreIndex to verify.

    Returns:
        True if all embeddings are valid, False otherwise.
    """
    try:
        docstore = index.docstore
        node_ids = list(docstore.docs.keys())

        if not node_ids:
            logger.warning("No nodes found in the index.")
            return False

        vector_store = index.vector_store
        all_valid = True

        for node_id in node_ids:
            embedding = vector_store.get(node_id)
            if embedding is None:
                logger.warning("Node %s is missing an embedding.", node_id)
                all_valid = False

        if all_valid:
            logger.info("All %d embeddings verified successfully.", len(node_ids))
        return all_valid
    except Exception as e:
        logger.error("Error verifying embeddings: %s", e)
        return False