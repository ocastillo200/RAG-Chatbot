import gradio as gr
import subprocess
import json

def query_model(prompt, history=None):
    """
    Query the Ollama model with a prompt
    
    Parameters:
    prompt (str): User's question
    history (list): Chat history
    
    Returns:
    str: Model's response
    """
    cmd = ["ollama", "run", "municipal-assistant", prompt, "--format", "json"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        try:
            response_json = json.loads(result.stdout)
            return response_json.get("response", "No response received")
        except json.JSONDecodeError:
            return result.stdout
    else:
        return f"Error: {result.stderr}"

# Create a Gradio interface
def create_chatbot_interface():
    with gr.Blocks() as demo:
        gr.Markdown("# My Custom Chatbot")
        chatbot = gr.Chatbot()
        msg = gr.Textbox(placeholder="Ask me something...")
        clear = gr.Button("Clear")

        def respond(message, chat_history):
            bot_message = query_model(message)
            chat_history.append((message, bot_message))
            return "", chat_history

        msg.submit(respond, [msg, chatbot], [msg, chatbot])
        clear.click(lambda: None, None, chatbot, queue=False)

    demo.launch(share=False)
    print("Chatbot interface is running at http://127.0.0.1:7860")

if __name__ == "__main__":
    create_chatbot_interface()