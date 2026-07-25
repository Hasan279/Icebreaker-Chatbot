"""Profile data chunking, embedding, and vector-store indexing."""

import json
import logging
from typing import Dict, List, Any, Optional

from llama_index.core import Document, VectorStoreIndex, Settings
from llama_index.core.node_parser import SentenceSplitter

from modules.llm_interface import create_ollama_embedding
import config

logger = logging.getLogger(__name__)


def split_profile_data(profile_data: Dict[str, Any]) -> List:
    """Convert profile JSON into chunked document nodes."""
    logger.info("Splitting profile data into nodes...")
    profile_text = json.dumps(profile_data, indent=2)
    document = Document(text=profile_text)

    splitter = SentenceSplitter(chunk_size=config.CHUNK_SIZE)
    nodes = splitter.get_nodes_from_documents([document])

    logger.info("Split profile into %d nodes", len(nodes))
    return nodes


def create_vector_database(nodes: List) -> Optional[VectorStoreIndex]:
    """Embed nodes and build an in-memory VectorStoreIndex."""
    try:
        logger.info("Creating vector database with %d nodes...", len(nodes))
        embed_model = create_ollama_embedding()
        Settings.embed_model = embed_model

        index = VectorStoreIndex(nodes, embed_model=embed_model, show_progress=True)
        logger.info("Vector database created successfully.")
        return index
    except Exception as e:
        logger.error("Failed to create vector database: %s", e)
        return None


def verify_embeddings(index: VectorStoreIndex) -> bool:
    """Check that every node in the index has an embedding."""
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