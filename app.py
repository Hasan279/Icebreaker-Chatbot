"""Gradio web interface for the Icebreaker Bot."""

import sys
import logging
import uuid
import gradio as gr

from modules.data_extraction import extract_linkedin_profile
from modules.data_processing import split_profile_data, create_vector_database, verify_embeddings
from modules.llm_interface import change_llm_model
from modules.query_engine import generate_initial_facts, answer_user_query
import config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(stream=sys.stdout)],
)

logger = logging.getLogger(__name__)

active_indices = {}


def process_profile(linkedin_url, api_key, use_mock, selected_model):
    """Process a LinkedIn profile and return initial facts plus a session ID."""
    if not use_mock and not linkedin_url:
        return "⚠️ Please enter a LinkedIn profile URL or select 'Use Mock Data'.", None
    if not use_mock and not api_key:
        return "⚠️ Please enter a ProxyCurl API key or select 'Use Mock Data'.", None

    if selected_model != config.LLM_MODEL_ID:
        change_llm_model(selected_model)

    try:
        effective_url = linkedin_url or "https://www.linkedin.com/in/leonkatsnelson/"
        profile_data = extract_linkedin_profile(effective_url, api_key=api_key, mock=use_mock)
        if not profile_data:
            return "❌ Failed to extract profile data.", None

        nodes = split_profile_data(profile_data)
        if not nodes:
            return "❌ Failed to split profile into chunks.", None

        index = create_vector_database(nodes)
        if index is None:
            return "❌ Failed to build vector database.", None

        verify_embeddings(index)
        facts = generate_initial_facts(index)

        session_id = str(uuid.uuid4())
        active_indices[session_id] = index

        return facts, session_id

    except Exception as e:
        logger.exception("Error processing profile")
        return f"❌ Error: {e}", None


def chat_with_profile(session_id, user_query, chat_history):
    """Answer a follow-up question against the stored profile index."""
    if not session_id or session_id not in active_indices:
        return chat_history + [[user_query, "⚠️ No profile loaded. Please process a LinkedIn profile first."]]

    if not user_query or not user_query.strip():
        return chat_history + [["", "⚠️ Please enter a question."]]

    try:
        index = active_indices[session_id]
        response = answer_user_query(index, user_query)
        return chat_history + [[user_query, str(response)]]
    except Exception as e:
        logger.exception("Error answering query")
        return chat_history + [[user_query, f"❌ Error: {e}"]]


def create_gradio_interface():
    """Build and return the Gradio Blocks UI."""
    available_models = ["phi3.5", "llama3", "mistral", "gemma2"]

    with gr.Blocks(title="LinkedIn Icebreaker Bot") as demo:
        gr.Markdown("# 🤝 LinkedIn Icebreaker Bot")
        gr.Markdown("Generate personalized icebreakers and chat about LinkedIn profiles — powered by **Ollama** running locally.")

        with gr.Tab("Process LinkedIn Profile"):
            with gr.Row():
                with gr.Column():
                    linkedin_url = gr.Textbox(
                        label="LinkedIn Profile URL",
                        placeholder="https://www.linkedin.com/in/username/",
                    )
                    api_key = gr.Textbox(
                        label="ProxyCurl API Key (leave empty when using mock data)",
                        placeholder="Your ProxyCurl API Key",
                        type="password",
                    )
                    use_mock = gr.Checkbox(label="Use Mock Data", value=True)
                    model_dropdown = gr.Dropdown(
                        choices=available_models,
                        label="Select Ollama Model",
                        value=config.LLM_MODEL_ID,
                    )
                    process_btn = gr.Button("🚀 Process Profile", variant="primary")

                with gr.Column():
                    result_text = gr.Textbox(label="Initial Facts", lines=10)
                    session_id = gr.Textbox(label="Session ID", visible=False)

            process_btn.click(
                fn=process_profile,
                inputs=[linkedin_url, api_key, use_mock, model_dropdown],
                outputs=[result_text, session_id],
            )

        with gr.Tab("Chat"):
            gr.Markdown("Chat with the processed LinkedIn profile")

            chatbot = gr.Chatbot(height=500)
            chat_input = gr.Textbox(
                label="Ask a question about the profile",
                placeholder="What is this person's current job title?",
            )

            chat_btn = gr.Button("Send 💬", variant="primary")

            chat_btn.click(
                fn=chat_with_profile,
                inputs=[session_id, chat_input, chatbot],
                outputs=[chatbot],
            )

            chat_input.submit(
                fn=chat_with_profile,
                inputs=[session_id, chat_input, chatbot],
                outputs=[chatbot],
            )

    return demo


if __name__ == "__main__":
    demo = create_gradio_interface()
    demo.launch(server_name="127.0.0.1", server_port=5000, share=True)
