import json
import os

def prepare_training_data(raw_data_folder, output_file):
    """
    Convert raw data files into a format suitable for Ollama fine-tuning.
    
    This function supports both .txt files (with "question: answer" pairs)
    and .json files that follow a structure with a "conversations" key,
    where each conversation is a list of messages containing 'role' and 'content'.
    
    Parameters:
    raw_data_folder (str): Folder containing your raw data files.
    output_file (str): Path to save the formatted training data.
    """
    formatted_data = []
    print(f"Processing raw data from {raw_data_folder}...")
    
    # Walk through all files in the specified folder
    for root, _, files in os.walk(raw_data_folder):
        for file in files:
            file_path = os.path.join(root, file)
            
            # Process JSON files
            if file.endswith('.json'):
                with open(file_path, 'r', encoding='utf-8') as f:
                    try:
                        data = json.load(f)
                    except json.JSONDecodeError as e:
                        print(f"Error decoding {file_path}: {e}")
                        continue

                # Expect the JSON to have a "conversations" key
                if "conversations" in data:
                    for conv in data["conversations"]:
                        # Each conv is a list of messages
                        # Iterate through messages and pair user and assistant turns
                        i = 0
                        while i < len(conv):
                            message = conv[i]
                            if message.get("role") == "user":
                                # Look ahead for the next assistant message
                                if i + 1 < len(conv) and conv[i + 1].get("role") == "assistant":
                                    prompt = message.get("content", "").strip()
                                    response = conv[i + 1].get("content", "").strip()
                                    formatted_data.append({
                                        "prompt": prompt,
                                        "response": response
                                    })
                                    i += 2
                                else:
                                    i += 1
                            else:
                                i += 1
                else:
                    print(f"Skipping {file_path}: missing 'conversations' key.")

            # Process TXT files (legacy format)
            elif file.endswith('.txt'):
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                # Assume pairs are separated by double newlines and a colon separates question and answer
                pairs = content.split('\n\n')
                for pair in pairs:
                    if ':' in pair:
                        question, answer = pair.split(':', 1)
                        formatted_data.append({
                            "prompt": question.strip(),
                            "response": answer.strip()
                        })

    # Write the formatted data to a JSON file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(formatted_data, f, indent=2)
    
    print(f"Processed {len(formatted_data)} question-answer pairs")
    return formatted_data

# Example usage
if __name__ == "__main__":
    prepare_training_data("./data", "./data/training_data.json")
