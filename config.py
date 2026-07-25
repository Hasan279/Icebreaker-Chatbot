"""Configuration for the Icebreaker Bot."""

OLLAMA_BASE_URL = "http://localhost:11434"

LLM_MODEL_ID = "phi3.5"
EMBEDDING_MODEL_ID = "BAAI/bge-small-en-v1.5"

HUGGINGFACE_API_TOKEN = ""

PROXYCURL_API_KEY = ""

MOCK_DATA_URL = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/ZRe59Y_NJyn3hZgnF1iFYA/linkedin-profile-data.json"

SIMILARITY_TOP_K = 5
TEMPERATURE = 0.0
MAX_NEW_TOKENS = 500
CONTEXT_WINDOW = 4096
MIN_NEW_TOKENS = 1
TOP_K = 50
TOP_P = 1

CHUNK_SIZE = 500

INITIAL_FACTS_TEMPLATE = """
You are an AI assistant that provides detailed answers based on the provided context.

Context information is below:

{context_str}

Based on the context provided, list 3 interesting facts about this person's career or education.

Answer in detail, using only the information provided in the context.
"""

USER_QUESTION_TEMPLATE = """
You are an AI assistant that provides detailed answers to questions based on the provided context.

Context information is below:

{context_str}

Question: {query_str}

Answer in full details, using only the information provided in the context. If the answer is not available in the context, say "I don't know. The information is not available on the LinkedIn page."
"""
