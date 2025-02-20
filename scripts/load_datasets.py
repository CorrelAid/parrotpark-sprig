import requests
import pandas as pd
import os
import json  # Import the json module for parsing JSON

# Repository base information
repo_base = [
    {
        "name": "arc",
        "base": "https://huggingface.co/datasets/openGPT-X/arcx/resolve/main",
        "vars": [
            "/arc_challenge_DE_test.jsonl",
            "/arc_challenge_DE_train.jsonl",
            "/arc_challenge_DE_validation.jsonl"
        ]
    },
    {
        "name": "hellaswag",
        "base": "https://huggingface.co/datasets/openGPT-X/hellaswagx/resolve/main",
        "vars": [
            "/hellaswag_DE_train.jsonl",
            "/hellaswag_DE_validation.jsonl"
        ]
    },
    {
        "name": "truthfulqa",
        "base": "https://huggingface.co/datasets/openGPT-X/truthfulqax/resolve/main",
        "vars": [
            "/truthfulqa_gen_DE_validation.jsonl"
        ]
    },
    {
        "name": "gsm8k",
        "base": "https://huggingface.co/datasets/openGPT-X/gsm8kx/resolve/main",
        "vars": [
            "/gsm8k_DE_test.jsonl",
            "/gsm8k_DE_train.jsonl",
        ]
    }
]

# Output directory for saving the combined CSV file
output_dir = "data/benchmark"

def download_and_combine_to_csv(name, base_url, file_paths, output_dir):
    # Create a subfolder for the dataset
    subfolder = os.path.join(output_dir, name)
    os.makedirs(subfolder, exist_ok=True)  # Ensure the subfolder exists

    combined_data = []  # List to store all data from the JSONL files

    for file_path in file_paths:
        # Construct the full URL
        url = f"{base_url}{file_path}"
        print(f"Downloading: {url}")

        # Download the JSONL file
        response = requests.get(url)
        if response.status_code == 200:
            # Parse the JSONL content
            jsonl_content = response.text.splitlines()
            data = [json.loads(line) for line in jsonl_content]  # Use json.loads to parse each line
            combined_data.extend(data)  # Append the data to the combined list
            print(f"Downloaded and processed: {file_path}")
        else:
            print(f"Failed to download: {url} (Status code: {response.status_code})")

    # Convert the combined data to a DataFrame
    df = pd.DataFrame(combined_data)

    # Save the combined DataFrame as a single CSV file
    output_file = os.path.join(subfolder, f"{name}_gptx.csv")
    df.to_csv(output_file, index=False)
    print(f"Saved combined CSV: {output_file}")

# Process each repository base
for repo in repo_base:
    download_and_combine_to_csv(repo["name"], repo["base"], repo["vars"], output_dir)
