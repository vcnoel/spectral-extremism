import pandas as pd
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.linear_model import LinearRegression
import json
from tqdm import tqdm
import os

def load_text_list(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    neutral = [d for d in data if d['category'] == 'riabi_immigrants_en_neutral']
    radical = [d for d in data if d['category'] == 'riabi_immigrants_en_radical']
    
    # Exact same sorting as 01_extract_riabi.py
    neutral = sorted(neutral, key=lambda x: len(x['text'].split()))
    radical = sorted(radical, key=lambda x: len(x['text'].split()))
    
    start_idx = 10
    subset_n = neutral[start_idx:start_idx+100]
    subset_r = radical[start_idx:start_idx+100]
    
    # Order matched 01_extract_riabi.py: 100 neutral then 100 radical
    return [d['text'] for d in subset_n] + [d['text'] for d in subset_r]

def calculate_perplexity(model, tokenizer, text):
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    input_ids = inputs["input_ids"]
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss
    return loss.item() # This is log-perplexity (mean cross-entropy)

def main():
    csv_path = 'data/riabi_features_N200.csv'
    dataset_path = 'data/extremism_dataset.json'
    
    df = pd.read_csv(csv_path)
    text_list = load_text_list(dataset_path)
    
    if len(df) != len(text_list):
        print(f"Warning: mismatch between CSV ({len(df)}) and text list ({len(text_list)})")
        # Truncate to the smaller one
        min_len = min(len(df), len(text_list))
        df = df.iloc[:min_len]
        text_list = text_list[:min_len]

    model_id = "meta-llama/Llama-3.2-3B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")
    model.eval()

    perplexities = []
    lengths = []
    
    print("Calculating Perplexity and Length for N=200...")
    for text in tqdm(text_list, desc="Processing samples"):
        lp = calculate_perplexity(model, tokenizer, text)
        length = len(tokenizer.encode(text))
        
        perplexities.append(lp)
        lengths.append(length)
        
    df['log_perplexity'] = perplexities
    df['length'] = lengths
    
    # Regression Analysis
    y = (df['label'] == 'Hate Speech').astype(int).values
    topo_features = ['L24_gini', 'L0_smoothness']
    control_features = ['length', 'log_perplexity']
    
    def get_sse(X, y):
        mod = LinearRegression()
        mod.fit(X, y)
        y_pred = mod.predict(X)
        return np.sum((y - y_pred)**2)

    sse_full = get_sse(df[topo_features + control_features].values, y)
    sse_reduced = get_sse(df[control_features].values, y)
    
    # Partial R-squared: (SSE_reduced - SSE_full) / SSE_reduced
    partial_r2 = (sse_reduced - sse_full) / sse_reduced if sse_reduced != 0 else 0
    
    print(f"\nCONFOUNDER_CONTROL_RESULTS:")
    print(f"Full Model SSE: {sse_full:.4f}")
    print(f"Reduced Model (Length+PPL) SSE: {sse_reduced:.4f}")
    print(f"Partial R-squared (Topological Features): {partial_r2:.6f}")
    
    output_dir = 'results/forensic'
    os.makedirs(output_dir, exist_ok=True)
    df.to_csv('data/riabi_features_enriched_N200.csv', index=False)
    print(f"Enriched dataset saved to data/riabi_features_enriched_N200.csv")

if __name__ == "__main__":
    main()
