import pandas as pd
import numpy as np
import torch
from spectral_trust import GSPDiagnosticsFramework, GSPConfig, SpectralAnalyzer
from transformers import BitsAndBytesConfig, AutoTokenizer, AutoModelForCausalLM
import json
from tqdm import tqdm
import os
import argparse

def load_riabi_pairs(path, n_pairs=20):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    neutral = [d for d in data if d['category'] == 'riabi_immigrants_en_neutral']
    radical = [d for d in data if d['category'] == 'riabi_immigrants_en_radical']
    
    neutral = sorted(neutral, key=lambda x: len(x['text'].split()))
    radical = sorted(radical, key=lambda x: len(x['text'].split()))
    
    start_idx = 10
    subset_n = neutral[start_idx:start_idx+n_pairs]
    subset_r = radical[start_idx:start_idx+n_pairs]
    
    return subset_n, subset_r

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_pairs", type=int, default=20)
    args = parser.parse_args()
    
    dataset_path = 'data/extremism_dataset.json'
    n_n, n_r = load_riabi_pairs(dataset_path, n_pairs=args.n_pairs)
    
    # Pivoting to Qwen-7B which is fully cached
    model_id = "Qwen/Qwen2.5-7B-Instruct"
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    
    config = GSPConfig(
        model_name=model_id,
        device="cuda",
        display_plots=False,
        save_intermediate=False,
        model_kwargs={"quantization_config": bnb_config}
    )
    
    results = []
    
    with GSPDiagnosticsFramework(config) as framework:
        print(f"Loading model {model_id}...", flush=True)
        framework.instrumenter.load_model(model_id)
        print("Model loaded successfully.", flush=True)
        
        # Process Neutral
        for s in tqdm(n_n, desc="Validation: Neutral"):
            print(f"Processing Neutral: {s['text'][:50]}...", flush=True)
            res = framework.analyze_text(s['text'])
            # Qwen has 28 layers (0-27). Using late layers: 20-27.
            ginis = [d.attention_gini for d in res['layer_diagnostics'] if 20 <= d.layer <= 27]
            results.append({'label': 'Neutral', 'avg_gini': np.mean(ginis) if ginis else 0})
            
        # Process Hate
        for s in tqdm(n_r, desc="Validation: Hate"):
            print(f"Processing Hate: {s['text'][:50]}...", flush=True)
            res = framework.analyze_text(s['text'])
            ginis = [d.attention_gini for d in res['layer_diagnostics'] if 20 <= d.layer <= 27]
            results.append({'label': 'Hate Speech', 'avg_gini': np.mean(ginis) if ginis else 0})
            
    df = pd.DataFrame(results)
    
    mean_n = df[df['label'] == 'Neutral']['avg_gini'].mean()
    mean_h = df[df['label'] == 'Hate Speech']['avg_gini'].mean()
    gini_delta = mean_n - mean_h
    
    print(f"\nQWEN_INVARIANCE_RESULTS:")
    print(f"Neutral Avg Gini (L20-27): {mean_n:.6f}")
    print(f"Hate Avg Gini (L20-27): {mean_h:.6f}")
    print(f"Qwen Gini Delta: {gini_delta:.6f}")
    
    if gini_delta > 0:
        print("Conclusion: Qwen confirmed original hypothesis—Hate Speech shows topological sparsity collapse.")
    else:
        print("Conclusion: Qwen does NOT show the same sparsity collapse pattern.")

    os.makedirs('results/forensic', exist_ok=True)
    df.to_csv('results/forensic/qwen_invariance_study.csv', index=False)

if __name__ == "__main__":
    main()
