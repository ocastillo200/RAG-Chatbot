import torch
import ollama
import os
import sys
from openai import OpenAI
import json
import argparse
from flask import Flask, request, jsonify
from flask_cors import CORS

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# ANSI escape codes for colors
PINK = '\033[95m'
CYAN = '\033[96m'
YELLOW = '\033[93m'
NEON_GREEN = '\033[92m'
RESET_COLOR = '\033[0m'

# Global variables to be initialized
client = None
vault_content = []
vault_embeddings_tensor = None
conversation_history = []
system_message = """Eres un asistente experto en proporcionar respuestas concisas sobre el documento adjunto en español. 
Tus objetivos son:
- Responder brevemente y con claridad
- Identificar y resaltar los departamentos o áreas relevantes mencionados en el contexto
- Extraer la información más importante
- Guiar al usuario hacia los departamentos específicos que pueden ayudarle mejor
Al final de tu respuesta, incluye una sección de "Departamento" que enumere los departamentos, direcciones o secciones relacionados a la consulta."""

def get_relevant_context(rewritten_input, vault_embeddings, vault_content, top_k=3):
    if vault_embeddings.nelement() == 0:  # Check if the tensor has any elements
        return []
    # Encode the rewritten input
    input_embedding = ollama.embeddings(model='mxbai-embed-large', prompt=rewritten_input)["embedding"]
    # Compute cosine similarity between the input and vault embeddings
    cos_scores = torch.cosine_similarity(torch.tensor(input_embedding).unsqueeze(0), vault_embeddings)
    # Adjust top_k if it's greater than the number of available scores
    top_k = min(top_k, len(cos_scores))
    # Sort the scores and get the top-k indices
    top_indices = torch.topk(cos_scores, k=top_k)[1].tolist()
    # Get the corresponding context from the vault
    relevant_context = [vault_content[idx].strip() for idx in top_indices]
    return relevant_context

def rewrite_query(user_input, conversation_history, ollama_model):
    context = "\n".join([f"{msg['role']}: {msg['content']}" for msg in conversation_history[-2:]])
    prompt = f"""Rewrite the following query by incorporating relevant context from the conversation history.
    The rewritten query should:
    
    - Preserve the core intent and meaning of the original query
    - Expand and clarify the query to make it more specific and informative for retrieving relevant context
    - Avoid introducing new topics or queries that deviate from the original query
    - DONT EVER ANSWER the Original query, but instead focus on rephrasing and expanding it into a new query
    
    Return ONLY the rewritten query text, without any additional formatting or explanations.
    
    Conversation History:
    {context}
    
    Original query: [{user_input}]
    
    Rewritten query: 
    """
    response = client.chat.completions.create(
        model=ollama_model,
        messages=[{"role": "system", "content": prompt}],
        max_tokens=200,
        n=1,
        temperature=0.1,
    )
    return response.choices[0].message.content.strip()

def ollama_chat_processing(user_input, ollama_model='llama3:latest'):
    global conversation_history, vault_embeddings_tensor, vault_content
    
    conversation_history.append({"role": "user", "content": user_input})
    
    if len(conversation_history) > 1:
        rewritten_query = rewrite_query(user_input, conversation_history, ollama_model)
        print(f"{PINK}Original Query: {user_input}{RESET_COLOR}")
        print(f"{PINK}Rewritten Query: {rewritten_query}{RESET_COLOR}")
    else:
        rewritten_query = user_input
    
    relevant_context = get_relevant_context(rewritten_query, vault_embeddings_tensor, vault_content)
    if relevant_context:
        context_str = "\n".join(relevant_context)
        print(f"Context Pulled from Documents: \n\n{CYAN}{context_str}{RESET_COLOR}")
    else:
        print(f"{CYAN}No relevant context found.{RESET_COLOR}")
    
    user_input_with_context = user_input
    if relevant_context:
        user_input_with_context = user_input + "\n\nRelevant Context:\n" + context_str
    
    conversation_history[-1]["content"] = user_input_with_context
    
    messages = [
        {"role": "system", "content": system_message},
        *conversation_history
    ]
    
    response = client.chat.completions.create(
        model=ollama_model,
        messages=messages,
        max_tokens=2000,
    )
    
    conversation_history.append({"role": "assistant", "content": response.choices[0].message.content})
    
    return response.choices[0].message.content

@app.route('/initialize', methods=['POST'])
def initialize_server():
    global client, vault_content, vault_embeddings_tensor

    # Configuration for the Ollama API client
    client = OpenAI(
        base_url='http://localhost:11434/v1',
        api_key='llama3'
    )

    # Load the vault content
    if os.path.exists("vault.txt"):
        with open("vault.txt", "r", encoding='utf-8') as vault_file:
            vault_content = vault_file.readlines()

    # Generate embeddings for the vault content using Ollama
    vault_embeddings = []
    for content in vault_content:
        response = ollama.embeddings(model='mxbai-embed-large', prompt=content)
        vault_embeddings.append(response["embedding"])

    # Convert to tensor
    vault_embeddings_tensor = torch.tensor(vault_embeddings)
    
    return jsonify({"status": "Server initialized successfully", "vault_size": len(vault_content)})

@app.route('/chat', methods=['POST'])
def chat():
    try:
        # Parse the incoming request data
        data = request.json
        
        # Extract messages from the request
        user_input = data.get('messages', [])[-1]['content']
        
        # Select the model (default to llama3:latest if not specified)
        model = data.get('model', 'llama3:latest')
        
        # Generate a response using the existing chat processing logic
        response_text = ollama_chat_processing(user_input, model)
        
        # Return the AI's response
        return jsonify({
            'message': {
                'content': response_text
            }
        })
    
    except Exception as e:
        print(f"Error processing chat request: {e}")
        return jsonify({
            'error': 'Failed to process chat request',
            'details': str(e)
        }), 500

if __name__ == '__main__':
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Ollama Chat Server")
    parser.add_argument("--port", type=int, default=5000, help="Port to run the server on")
    args = parser.parse_args()

    # Run the Flask app
    app.run(host='0.0.0.0', port=args.port, debug=True)