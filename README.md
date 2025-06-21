# ChatBotLLM

Chatbot trained to answer specific requests about a document using PyTorch and RAG (Retrieval Augmented Generation) technique.

**Prerequisites:**

1.  **Install Ollama:** You need to install Ollama previously from their official website: [https://ollama.com/](https://ollama.com/)
2.  **Download a Local Model:** Download the LLM model you want to run locally using Ollama.

**Setup:**

```bash
pip3 install -r requirements.txt

```

**Usage:**

1. upload.py: This script takes a PDF file, cleans it, and transforms it into a vault.txt file. This vault.txt file is used to feed information to the LLM for the RAG process.
2. server.py: This is the main application file. It uses the OpenAI API, runs the selected LLM model locally (via Ollama) and create a Flask server if you want to make requests from an App.
