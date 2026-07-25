"""CLI entry point for the Icebreaker Bot."""

import sys
import logging
import argparse

from modules.data_extraction import extract_linkedin_profile
from modules.data_processing import split_profile_data, create_vector_database, verify_embeddings
from modules.query_engine import generate_initial_facts, answer_user_query
import config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(stream=sys.stdout)],
)

logger = logging.getLogger(__name__)


def process_linkedin(linkedin_url, api_key=None, mock=False):
    """Run the full pipeline: extract → chunk → embed → generate facts → chat."""
    print("\n[*] Extracting LinkedIn profile data...")
    profile_data = extract_linkedin_profile(linkedin_url, api_key=api_key, mock=mock)
    if not profile_data:
        print("[ERROR] Failed to extract profile data. Exiting.")
        return

    print("[*] Splitting profile data into chunks...")
    nodes = split_profile_data(profile_data)
    if not nodes:
        print("[ERROR] Failed to split profile data. Exiting.")
        return

    print("[*] Building vector database (this may take a moment)...")
    index = create_vector_database(nodes)
    if index is None:
        print("[ERROR] Failed to create vector database. Exiting.")
        return

    print("[*] Verifying embeddings...")
    verify_embeddings(index)

    print("\n[*] Generating interesting facts about the profile...\n")
    facts = generate_initial_facts(index)
    print("=" * 60)
    print(facts)
    print("=" * 60)

    chatbot_interface(index)


def chatbot_interface(index):
    """Interactive REPL for asking questions about the profile."""
    print("\n[Icebreaker Chatbot] Ready! Type 'exit', 'quit', or 'bye' to stop.\n")
    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in {"exit", "quit", "bye"}:
            print("Goodbye!")
            break

        print("Thinking...\n")
        response = answer_user_query(index, user_input)
        print(f"Bot: {response}\n")


def main():
    """Parse CLI arguments and launch the pipeline."""
    parser = argparse.ArgumentParser(description='Icebreaker Bot - LinkedIn Profile Analyzer')
    parser.add_argument('--url', type=str, help='LinkedIn profile URL')
    parser.add_argument('--api-key', type=str, help='ProxyCurl API key')
    parser.add_argument('--mock', action='store_true', help='Use mock data instead of API')
    parser.add_argument('--model', type=str, help='Ollama model to use (e.g., "phi3.5", "llama3")')

    args = parser.parse_args()

    if args.model:
        from modules.llm_interface import change_llm_model
        change_llm_model(args.model)

    if args.mock:
        linkedin_url = args.url or ""
        use_mock = True
    else:
        linkedin_url = args.url or input("Enter LinkedIn profile URL (or press Enter to use mock data): ")
        use_mock = not linkedin_url

    api_key = args.api_key or config.PROXYCURL_API_KEY

    if not use_mock and not api_key:
        api_key = input("Enter ProxyCurl API key: ")

    if use_mock and not linkedin_url:
        linkedin_url = "https://www.linkedin.com/in/leonkatsnelson/"

    process_linkedin(linkedin_url, api_key, mock=use_mock)


if __name__ == "__main__":
    main()