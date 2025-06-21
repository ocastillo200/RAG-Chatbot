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

1. `upload.py`: This script takes a PDF file, cleans it, and transforms it into a `vault.txt` file. This file is used to feed information to the LLM for the RAG process.
2. `server.py`: This is the main application file. It uses the OpenAI API, runs the selected LLM model locally (via Ollama) and create a Flask server to make requests from an application.

## Configuration Adjustments for `server.py`

---

1. Modify the system message to align with your specific use case. This helps guide the LLM's overall behavior and persona.

2. Redefine the prompt within the `rewrite_query` function.

3. Adjust the `top-k` parameter in `get_relevant_context` function

* **Higher `top-k` values**: Your model will incorporate more contextual information. This can lead to more accurate and comprehensive answers, but it will generally **increase the response time**.
* **Lower `top-k` values**: Your model will prioritize speed by limiting the context it considers. This is ideal for applications where **real-time responses** are critical, though it might occasionally result in less detailed answers.

---
