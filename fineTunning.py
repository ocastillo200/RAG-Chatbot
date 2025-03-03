import json
import subprocess
import os

def fine_tune_model(base_model, new_model_name, training_data_file, modelfile_path=None):
    """
    Fine-tune an Ollama model with custom training data
    
    Parameters:
    base_model (str): Name of the base model to build upon (e.g. "deepseek-r1:7b")
    new_model_name (str): Name for your fine-tuned model
    training_data_file (str): Path to your formatted training data
    modelfile_path (str, optional): Path to save the Modelfile
    """
    # Load the training data
    with open(training_data_file, 'r', encoding='utf-8') as f:
        training_data = json.load(f)
    
    print(f"Preparing to fine-tune with {len(training_data)} examples...")
    
    # Create a directory for the Modelfile if not provided
    if not modelfile_path:
        modelfile_path = f"./{new_model_name}"
        os.makedirs(modelfile_path, exist_ok=True)
    
    # Create the Modelfile with correct syntax
    modelfile_content = f"""FROM {base_model}
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER stop "Human:"

# System prompt
SYSTEM "Eres un asistente municipal que ayuda a los ciudadanos a identificar qué departamento municipal puede atender sus consultas o resolver sus problemas. Proporciona respuestas claras y precisas, mencionando específicamente la dirección o departamento municipal al que deben acudir."

# Template for conversation format
TEMPLATE \"\"\"
Human: {{prompt}}
Assistant: {{response}}
\"\"\"
"""
    
    # Write the Modelfile
    modelfile_location = os.path.join(modelfile_path, "Modelfile")
    with open(modelfile_location, "w", encoding="utf-8") as f:
        f.write(modelfile_content)
    
    print(f"Created Modelfile at {modelfile_location}")
    
    # Create a training file with examples in the right format
    training_jsonl = os.path.join(modelfile_path, "training_data.jsonl")
    with open(training_jsonl, "w", encoding="utf-8") as f:
        for example in training_data:
            prompt = example["prompt"].replace('"', '\\"')  # Escape double quotes
            response = example["response"].replace('"', '\\"')
            f.write(json.dumps({"prompt": prompt, "response": response}) + "\n")
    
    print(f"Created training data file at {training_jsonl}")
    
    # First check if the base model is available
    try:
        check_model_cmd = f"ollama list"
        result = subprocess.run(check_model_cmd, shell=True, text=True, capture_output=True)
        if base_model not in result.stdout:
            print(f"Base model {base_model} not found. Pulling it now...")
            subprocess.run(f"ollama pull {base_model}", shell=True, text=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error checking or pulling base model: {e}")
        return False
    
    # Convert paths to be compatible with the command
    modelfile_path_for_cmd = os.path.normpath(os.path.join(modelfile_path, "Modelfile"))
    
    # Create the model using Ollama with more detailed error handling
    create_cmd = f"ollama create {new_model_name} -f {modelfile_path_for_cmd}"
    print(f"Running command: {create_cmd}")
    
    try:
        process = subprocess.run(create_cmd, shell=True, text=True, check=True, capture_output=True)
        print(f"Successfully created model: {new_model_name}")
        print(process.stdout)
        
        # Now train the model with examples
        print("Training model with examples...")
        for i, example in enumerate(training_data):
            prompt = example["prompt"].replace('"', '\\"').replace('\n', ' ')
            response = example["response"].replace('"', '\\"').replace('\n', ' ')

            print(f"Training with example {i+1}:")
            print(f"Prompt: {prompt}")
            print(f"Response: {response}")
            
            # Run the model with training data
            train_cmd = f'ollama run {new_model_name} "Human: {prompt}" "Assistant: {response}"'
            try:
                train_result = subprocess.run(train_cmd, shell=True, text=True, capture_output=True)
                if (i+1) % 5 == 0 or (i+1) == len(training_data):
                    print(f"Processed {i+1}/{len(training_data)} examples")
            except subprocess.TimeoutExpired:
                print(f"Example {i+1} timed out, continuing with next example")
        
        print(f"Training complete. Your model '{new_model_name}' is ready to use.")
    except subprocess.CalledProcessError as e:
        print(f"Error creating model: {e}")
        if hasattr(e, 'stderr') and e.stderr:
            print(f"Error details: {e.stderr}")
        else:
            # Try to get more detailed error information
            debug_cmd = f"ollama create {new_model_name} -f {modelfile_path_for_cmd} --verbose"
            try:
                debug_process = subprocess.run(debug_cmd, shell=True, text=True, capture_output=True)
                print(f"Debug output: {debug_process.stderr}")
            except Exception as debug_e:
                print(f"Debug attempt failed: {debug_e}")
        return False
    
    print("You can now use your model with:")
    print(f"ollama run {new_model_name}")
    
    return True

# Example usage
if __name__ == "__main__":
    fine_tune_model(
        base_model="deepseek-r1:7b", 
        new_model_name="municipal-assistant",
        training_data_file="./data/data.json"
    )