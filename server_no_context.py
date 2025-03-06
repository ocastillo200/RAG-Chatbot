import torch
import ollama
import os
import sys
import time
from openai import OpenAI
import json
import argparse
from flask import Flask, request, jsonify
from flask_cors import CORS

if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf8')(sys.stdout.buffer)
    sys.stderr = codecs.getwriter('utf8')(sys.stderr.buffer)

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# ANSI escape codes for colors
PINK = '\033[95m'
CYAN = '\033[96m'
YELLOW = '\033[93m'
NEON_GREEN = '\033[92m'
RED = '\033[91m'
BLUE = '\033[94m'
RESET_COLOR = '\033[0m'

# Global variables to be initialized
client = None
vault_content = []
vault_embeddings_tensor = None
conversation_history = []
system_message = """Eres un asistente experto en proporcionar respuestas concisas sobre el documento adjunto en español. 
Tus objetivos son:
- Prioriza responder la pregunta del usuario de manera clara y precisa
- Utiliza el contexto solo como referencia complementaria, no como respuesta principal
- Si el contexto es relevante, integra sutilmente la información en tu respuesta
- Identificar y resaltar los departamentos o áreas relevantes mencionados en el contexto
- Extraer la información más importante
- Cada respuesta que proporciones debe ser una instrucción del mapeo de la pregunta a un departamento, dirección o sección específica
- Guiar al usuario hacia los departamentos específicos que pueden ayudarle mejor
- Al final de tu respuesta, incluye una sección de "Departamento" que enumere los departamentos, direcciones o secciones relacionados a la consulta
- IMPORTANTE: No comiences tus respuestas con saludos o frases como "Hola", "Claro", "Por supuesto", etc. Responde directamente a la consulta."""

def get_relevant_context(query, vault_embeddings, vault_content, top_k=3):
    print(f"{YELLOW}[DEBUG] get_relevant_context() - Starting context retrieval{RESET_COLOR}")
    print(f"{YELLOW}[DEBUG] get_relevant_context() - Input query: {query}{RESET_COLOR}")
    print(f"{YELLOW}[DEBUG] get_relevant_context() - Requested top_k: {top_k}{RESET_COLOR}")
    
    if vault_embeddings.nelement() == 0:
        print(f"{RED}[DEBUG] get_relevant_context() - Empty vault embeddings tensor{RESET_COLOR}")
        return []
    
    print(f"{YELLOW}[DEBUG] get_relevant_context() - Generating embeddings for input{RESET_COLOR}")
    start_time = time.time()
    try:
        input_embedding_response = ollama.embeddings(model='mxbai-embed-large', prompt=query)
        input_embedding = input_embedding_response["embedding"]
        print(f"{YELLOW}[DEBUG] get_relevant_context() - Embedding generation took {time.time() - start_time:.2f} seconds{RESET_COLOR}")
    except Exception as e:
        print(f"{RED}[DEBUG] get_relevant_context() - Error generating embeddings: {str(e)}{RESET_COLOR}")
        return []
    
    print(f"{YELLOW}[DEBUG] get_relevant_context() - Computing cosine similarity with {len(vault_embeddings)} vault items{RESET_COLOR}")
    cos_scores = torch.cosine_similarity(torch.tensor(input_embedding).unsqueeze(0), vault_embeddings)
    
    top_k = min(top_k, len(cos_scores))
    print(f"{YELLOW}[DEBUG] get_relevant_context() - Adjusted top_k: {top_k}{RESET_COLOR}")
    
    top_indices = torch.topk(cos_scores, k=top_k)[1].tolist()
    top_scores = torch.topk(cos_scores, k=top_k)[0].tolist()
    
    for i, (idx, score) in enumerate(zip(top_indices, top_scores)):
        print(f"{YELLOW}[DEBUG] get_relevant_context() - Top {i+1}: index={idx}, score={score:.4f}{RESET_COLOR}")
    
    relevant_context = [vault_content[idx].strip() for idx in top_indices]
    print(f"{YELLOW}[DEBUG] get_relevant_context() - Retrieved {len(relevant_context)} context items{RESET_COLOR}")
    
    return relevant_context

def ollama_chat_processing(user_input, ollama_model='llama3:latest'):
    global conversation_history, vault_embeddings_tensor, vault_content
    
    print(f"{NEON_GREEN}[DEBUG] ollama_chat_processing() - Processing new request{RESET_COLOR}")
    print(f"{NEON_GREEN}[DEBUG] ollama_chat_processing() - User input: {user_input}{RESET_COLOR}")
    print(f"{NEON_GREEN}[DEBUG] ollama_chat_processing() - Using model: {ollama_model}{RESET_COLOR}")
    print(f"{NEON_GREEN}[DEBUG] ollama_chat_processing() - Conversation history length: {len(conversation_history)}{RESET_COLOR}")
    
    conversation_history.append({"role": "user", "content": user_input})
    
    # Use the original query directly without rewriting
    print(f"{PINK}Query: {user_input}{RESET_COLOR}")
    
    try:
        print(f"{NEON_GREEN}[DEBUG] ollama_chat_processing() - Getting relevant context{RESET_COLOR}")
        relevant_context = get_relevant_context(user_input, vault_embeddings_tensor, vault_content)
        if relevant_context:
            context_str = "\n".join(relevant_context)
            print(f"Context Pulled from Documents: \n\n{CYAN}{context_str}{RESET_COLOR}")
            print(f"{NEON_GREEN}[DEBUG] ollama_chat_processing() - Retrieved {len(relevant_context)} context items ({len(context_str)} chars){RESET_COLOR}")
        else:
            print(f"{CYAN}No relevant context found.{RESET_COLOR}")
            print(f"{NEON_GREEN}[DEBUG] ollama_chat_processing() - No relevant context found{RESET_COLOR}")
            context_str = ""
    except Exception as e:
        print(f"{RED}[DEBUG] ollama_chat_processing() - Error retrieving context: {str(e)}{RESET_COLOR}")
        relevant_context = []
        context_str = ""
    
    user_input_with_context = user_input
    if relevant_context:
        user_input_with_context = user_input + "\n\nRelevant Context:\n" + context_str
        print(f"{NEON_GREEN}[DEBUG] ollama_chat_processing() - Final input length with context: {len(user_input_with_context)} chars{RESET_COLOR}")
    
    conversation_history[-1]["content"] = user_input_with_context
    
    messages = [
        {"role": "system", "content": system_message},
        *conversation_history
    ]
    print(f"{NEON_GREEN}[DEBUG] ollama_chat_processing() - Sending {len(messages)} messages to model{RESET_COLOR}")
    
    start_time = time.time()
    try:
        response = client.chat.completions.create(
            model=ollama_model,
            messages=messages,
            max_tokens=2000,
        )
        response_content = response.choices[0].message.content
        print(f"{NEON_GREEN}[DEBUG] ollama_chat_processing() - Model response took {time.time() - start_time:.2f} seconds{RESET_COLOR}")
        print(f"{NEON_GREEN}[DEBUG] ollama_chat_processing() - Response length: {len(response_content)} chars{RESET_COLOR}")
    except Exception as e:
        print(f"{RED}[DEBUG] ollama_chat_processing() - Error getting model response: {str(e)}{RESET_COLOR}")
        response_content = f"Lo siento, hubo un error al procesar tu consulta: {str(e)}"
    
    conversation_history.append({"role": "assistant", "content": response_content})
    
    return response_content

@app.route('/')
def index():
    return "Ollama Chat Server", 200

@app.route('/chat', methods=['POST'])
def chat():
    print(f"{NEON_GREEN}[DEBUG] chat() - Received chat request{RESET_COLOR}")
    try:
        data = request.json
        print(f"{NEON_GREEN}[DEBUG] chat() - Request payload size: {len(str(data))} chars{RESET_COLOR}")
        
        if 'messages' not in data or not data['messages']:
            print(f"{RED}[DEBUG] chat() - No messages found in request{RESET_COLOR}")
            return jsonify({'error': 'No messages provided'}), 400
            
        user_input = data.get('messages', [])[-1]['content']
        print(f"{NEON_GREEN}[DEBUG] chat() - Extracted user input: {user_input}{RESET_COLOR}")
        
        model = data.get('model', 'llama3:latest')
        print(f"{NEON_GREEN}[DEBUG] chat() - Using model: {model}{RESET_COLOR}")
        
        if client is None:
            print(f"{RED}[DEBUG] chat() - OpenAI client not initialized.{RESET_COLOR}")
            return jsonify({'error': 'Server not initialized.'}), 500
            
        print(f"{NEON_GREEN}[DEBUG] chat() - Calling ollama_chat_processing(){RESET_COLOR}")
        start_time = time.time()
        response_text = ollama_chat_processing(user_input, model)
        processing_time = time.time() - start_time
        print(f"{NEON_GREEN}[DEBUG] chat() - Processing completed in {processing_time:.2f} seconds{RESET_COLOR}")
        
        return jsonify({
            'message': {
                'content': response_text
            }
        })
    
    except Exception as e:
        print(f"{RED}[DEBUG] chat() - Error processing chat request: {str(e)}{RESET_COLOR}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'error': 'Failed to process chat request',
            'details': str(e)
        }), 500

def initialize_server_globals():
    global client, vault_content, vault_embeddings_tensor
    print(f"{CYAN}[DEBUG] initialize_server_globals() - Starting server initialization{RESET_COLOR}")
    
    try:
        print(f"{CYAN}[DEBUG] initialize_server_globals() - Creating OpenAI client{RESET_COLOR}")
        client = OpenAI(
            base_url='http://localhost:11434/v1',
            api_key='llama3'
        )
        print(f"{CYAN}[DEBUG] initialize_server_globals() - OpenAI client created successfully{RESET_COLOR}")
    except Exception as e:
        print(f"{RED}[DEBUG] initialize_server_globals() - Error creating OpenAI client: {str(e)}{RESET_COLOR}")
        sys.exit(1)

    try:
        if os.path.exists("vault.txt"):
            print(f"{CYAN}[DEBUG] initialize_server_globals() - Loading vault content from vault.txt{RESET_COLOR}")
            with open("vault.txt", "r", encoding='utf-8') as vault_file:
                vault_content = vault_file.readlines()
            print(f"{CYAN}[DEBUG] initialize_server_globals() - Loaded {len(vault_content)} vault entries{RESET_COLOR}")
        else:
            print(f"{YELLOW}[DEBUG] initialize_server_globals() - vault.txt not found, initializing with empty vault{RESET_COLOR}")
            vault_content = []
    except Exception as e:
        print(f"{RED}[DEBUG] initialize_server_globals() - Error loading vault content: {str(e)}{RESET_COLOR}")
        sys.exit(1)

    try:
        print(f"{CYAN}[DEBUG] initialize_server_globals() - Generating embeddings for {len(vault_content)} vault items{RESET_COLOR}")
        vault_embeddings = []
        start_time = time.time()
        for i, content in enumerate(vault_content):
            if i % 10 == 0:
                print(f"{CYAN}[DEBUG] initialize_server_globals() - Processing embedding {i+1}/{len(vault_content)}{RESET_COLOR}")
            try:
                response = ollama.embeddings(model='mxbai-embed-large', prompt=content)
                vault_embeddings.append(response["embedding"])
            except Exception as e:
                print(f"{RED}[DEBUG] initialize_server_globals() - Error generating embedding for item {i}: {str(e)}{RESET_COLOR}")
                if vault_embeddings:
                    vault_embeddings.append([0.0] * len(vault_embeddings[0]))
                else:
                    print(f"{RED}[DEBUG] initialize_server_globals() - Cannot create fallback embedding with unknown dimensions{RESET_COLOR}")
                    sys.exit(1)
        embeddings_time = time.time() - start_time
        print(f"{CYAN}[DEBUG] initialize_server_globals() - Embeddings generation completed in {embeddings_time:.2f} seconds{RESET_COLOR}")
        
        vault_embeddings_tensor = torch.tensor(vault_embeddings)
        print(f"{CYAN}[DEBUG] initialize_server_globals() - Created tensor with shape {vault_embeddings_tensor.shape}{RESET_COLOR}")
    except Exception as e:
        print(f"{RED}[DEBUG] initialize_server_globals() - Error creating embeddings tensor: {str(e)}{RESET_COLOR}")
        sys.exit(1)
    
    print(f"{CYAN}[DEBUG] initialize_server_globals() - Server initialization complete{RESET_COLOR}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Ollama Chat Server")
    parser.add_argument("--port", type=int, default=5000, help="Port to run the server on")
    args = parser.parse_args()
    app.debug = True
    print(f"{CYAN}[DEBUG] main() - Starting Ollama Chat Server on port {args.port}{RESET_COLOR}")
    print(f"{CYAN}[DEBUG] main() - Debug mode is enabled{RESET_COLOR}")
    
    # Initialize globals before starting the server
    initialize_server_globals()
    
    app.run(host='0.0.0.0', port=args.port, debug=True, use_reloader=False)