"""Icebreaker Bot modules."""

from modules.data_extraction import extract_linkedin_profile
from modules.data_processing import split_profile_data, create_vector_database, verify_embeddings
from modules.llm_interface import create_huggingface_embedding, create_ollama_llm, change_llm_model
from modules.query_engine import generate_initial_facts, answer_user_query