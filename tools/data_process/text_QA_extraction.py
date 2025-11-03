# This file is for the QA text file to the json file
# The main idea is that, for the QA text, the question always ends with a question mark, and
# the answer is the text between next question
import os
import json
from config import BASE_DIR

# Directory containing the text files
directory_path = BASE_DIR / "data/example_articles"
output_path = BASE_DIR / "data/Web-FQA-HPV-clean/result/"

# Function to extract QA pairs from a text file
def extract_qa_pairs_from_text(text):
    lines = text.split('\n')
    question = ""
    answer = ""
    qa_pairs = []

    for line in lines:
        line = line.strip()
        if line.endswith("?"):
            if question and answer:
                qa_pairs.append((question, answer.strip()))
                answer = ""
            question = line
        else:
            answer += " " + line

    # Add the last QA pair
    if question and answer:
        qa_pairs.append((question, answer.strip()))

    return qa_pairs



# Iterate over each file in the directory
for file_name in os.listdir(directory_path):
    file_id = ""
    # List to store all QA pairs with unique IDs
    qa_data = []
    if file_name.endswith(".txt"):
        file_id = os.path.splitext(file_name)[0]  # Use file name without extension as file_id
        file_path = os.path.join(directory_path, file_name)
        
        with open(file_path, 'r') as file:
            text = file.read()
            qa_pairs = extract_qa_pairs_from_text(text)
        
        # Assign unique ID to each QA pair
        for index, (question, answer) in enumerate(qa_pairs, start=1):
            qa_id = f"{file_id}_{index}"  # Unique ID combining file identifier and index
            qa_data.append({
                "id": qa_id,
                "question": question,
                "answer": answer,
                "source":file_id
            })

    # Save all QA pairs to a JSON file
    with open(output_path + f'{file_id}.json', 'w') as json_file:
        json.dump(qa_data, json_file, indent=4)

    print(f"QA extraction and ID assignment of {file_id} completed successfully.")
