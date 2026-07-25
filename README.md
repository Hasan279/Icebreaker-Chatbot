# Icebreaker Chatbot

A RAG-powered chatbot that analyzes LinkedIn profiles and generates personalized icebreakers. It extracts profile data, builds a vector index, and answers questions about the person using retrieval-augmented generation.

## Architecture

```
LinkedIn URL ──► ProxyCurl API ──► Profile JSON ──► Chunking ──► Embeddings (HuggingFace API) ──► Vector Index
                                                                                                       │
                                                                                                       ▼
                                                                                            Query Engine (Ollama LLM)
                                                                                                       │
                                                                                                       ▼
                                                                                              Answer / Facts
```

## Tech Stack

| Component | Technology |
|-----------|------------|
| LLM | Ollama (phi3.5, llama3, mistral, gemma2) |
| Embeddings | HuggingFace Inference API (BAAI/bge-small-en-v1.5) |
| RAG Framework | LlamaIndex |
| Web UI | Gradio |
| Profile Data | ProxyCurl API or mock JSON |

## Setup

### Prerequisites

- [Ollama](https://ollama.ai/) installed and running with a model pulled:
  ```bash
  ollama pull phi3.5
  ```
- A [HuggingFace API token](https://huggingface.co/settings/tokens) with inference permissions
- (Optional) A [ProxyCurl API key](https://nubela.co/proxycurl/) for real LinkedIn data

### Installation

```bash
git clone https://github.com/Hasan279/Icebreaker-Chatbot.git
cd Icebreaker-Chatbot
python -m venv .venv
.venv\Scripts\activate       # Windows
pip install -r requirements.txt
```

### Configuration

Edit `config.py` and set your tokens:

```python
HUGGINGFACE_API_TOKEN = "hf_your_token_here"
PROXYCURL_API_KEY = "your_key_here"          # only needed for real profiles
```

## Usage

### Web Interface (Gradio)

```bash
python app.py
```

Opens at `http://127.0.0.1:5000`. Check **Use Mock Data**, click **Process Profile**, then switch to the **Chat** tab.

### CLI

```bash
python main.py --mock                        # mock data
python main.py --url "https://linkedin.com/in/someone" --api-key "KEY"
python main.py --mock --model llama3         # use a different Ollama model
```

## Project Structure

```
icebreaker/
├── app.py                     # Gradio web UI
├── main.py                    # CLI entry point
├── config.py                  # All configuration values
├── requirements.txt
├── modules/
│   ├── __init__.py
│   ├── data_extraction.py     # LinkedIn profile fetching
│   ├── data_processing.py     # Chunking & vector index
│   ├── llm_interface.py       # LLM & embedding setup
│   └── query_engine.py        # RAG query pipeline
└── README.md
```

## License

MIT
