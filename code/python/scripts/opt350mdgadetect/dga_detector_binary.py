"""
Description:
This script uses a pre-trained causal language model to calculate the probability scores
for specific tokens ('neg' and 'pos') given a context that is influenced by a domain.
The domains are read from a CSV file.

Key Functions:
- get_token_probability: Computes the probability of a specific token in the context provided.
- get_domain_probabilities: For a given domain, computes the probabilities of 'neg' and 'pos' tokens.
- main: Orchestrates the reading of domains from the CSV, computes their probabilities using the above functions, and optionally saves the results to a new CSV.

Usage:
To use this script, one must provide an input CSV path with a 'domain' column, a model path, and an optional output path for results.

Example Command:
python <script_name> --csv_path domains.csv --model_path /path/to/model/dir --output_path results.csv

Dependencies:
- torch: Deep learning library to work with PyTorch models.
- pandas: Data manipulation and analysis library.
- transformers: Library for state-of-the-art NLP models, here specifically for loading the pre-trained causal language model and tokenizer.

Note:
The script assumes the model and data reside on a CUDA-enabled GPU device.
"""

import torch
import pandas as pd
import argparse
from tqdm import tqdm
from transformers import (AutoModelForCausalLM, AutoTokenizer)

def get_token_probability(model, tokenizer, context, target_token):
    # Tokenize the input
    inputs = tokenizer(context, return_tensors='pt').to("cuda")
    
    # Get logits from the model
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits[:, -1, :]  # get the logits of the last token in the input
    
    # Convert logits to probabilities
    probabilities = torch.nn.functional.softmax(logits, dim=-1)
    
    # Get the index of the target token
    token_id = tokenizer.encode(target_token, add_special_tokens=False)[0]
    
    # Get the probability of the target token
    target_probability = probabilities[0, token_id].item()
    
    return target_probability

def get_domain_probabilities(model, tokenizer, domain):
    context = f"""#domain: {domain}\n#label: """
    prob_neg = get_token_probability(model, tokenizer, context, ' neg')
    prob_pos = get_token_probability(model, tokenizer, context, ' pos')
    return prob_neg, prob_pos

def main(args):
    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        local_files_only=True,
        device_map={"": 0},
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m", trust_remote_code=True)
    
    # Read domains from CSV
    df = pd.read_csv(args.csv_path)
    
    # Ensure the column with domains is named 'domain'
    assert 'domain' in df.columns, "CSV must have a 'domain' column"
    
    # Calculate probabilities and store in dataframe
    prob_neg_list = []
    prob_pos_list = []
    
    for domain in tqdm(df['domain'], desc="Processing domains"):
        prob_neg, prob_pos = get_domain_probabilities(model, tokenizer, domain)
        prob_neg_list.append(prob_neg)
        prob_pos_list.append(prob_pos)
    
    df['prob_neg'] = prob_neg_list
    df['prob_pos'] = prob_pos_list
    
    # Optionally, save results to new CSV
    df.to_csv(args.output_path, index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Calculate probabilities for domains in CSV")
    parser.add_argument("--csv_path", type=str, required=True, help="Path to input CSV containing domains.")
    parser.add_argument("--model_path", type=str, default="/home/harpo/CEPH/LLM-models/opt350-dga/", help="Path to model directory.")
    parser.add_argument("--output_path", type=str, default="output.csv", help="Path to save results CSV.")
    args = parser.parse_args()
    main(args)


