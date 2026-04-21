import json
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from spectral_trust import GSPDiagnosticsFramework, GSPConfig
import os

def load_riabi_data(path, n_per_class=100):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    neutral = [d for d in data if d['category'] == 'riabi_immigrants_en_neutral']
    radical = [d for d in data if d['category'] == 'riabi_immigrants_en_radical']
    
    # Simple word-count matching filter (ensure lengths are comparable)
    neutral = sorted(neutral, key=lambda x: len(x['text'].split()))
    radical = sorted(radical, key=lambda x: len(x['text'].split()))
    
    # Pick N=100 from each (using middle to avoid extreme outliers)
    start_idx = 10
    subset_n = neutral[start_idx:start_idx+n_per_class]
    subset_r = radical[start_idx:start_idx+n_per_class]
    
    samples = []
    for d in subset_n:
        d['label_str'] = 'Neutral'
        samples.append(d)
    for d in subset_r:
        d['label_str'] = 'Hate Speech'
        samples.append(d)
    
    return samples

def extract():
    dataset_path = 'data/extremism_dataset.json'
    samples = load_riabi_data(dataset_path)
    
    config = GSPConfig(
        model_name="meta-llama/Llama-3.2-3B-Instruct",
        device="cuda",
        display_plots=False,
        save_intermediate=False,
        calc_velocity=True
    )
    
    results_rows = []
    
    with GSPDiagnosticsFramework(config) as framework:
        framework.instrumenter.load_model("meta-llama/Llama-3.2-3B-Instruct")
        
        for sample in tqdm(samples, desc="Extracting Riabi Features"):
            # Enforce strict BOS masking via sub-graph indexing if we had the length
            # The analyze_text internally handles the graph construction.
            # We'll calculate velocity across layers.
            
            # Get prompt length for masking (Tokenize first)
            tokens = framework.instrumenter.tokenizer.encode(sample['text'], return_tensors='pt')
            n_tokens = tokens.shape[1]
            
            # Mask BOS (index 0) if it's the standard Llama-3 BOS
            # The user requested 1-{N-1}
            target_indices = list(range(1, n_tokens))
            
            res = framework.analyze_text(sample['text'], subgraph_indices=target_indices)
            
            row = {
                'sample_id': sample.get('id', 'unknown'),
                'label': sample['label_str'],
                'tokens': n_tokens,
                'max_velocity': res['velocity_metrics'].get('max_velocity_value'),
                'max_velocity_layer': res['velocity_metrics'].get('max_velocity_layer_index')
            }
            
            # Flatten layer diagnostics
            for diag in res['layer_diagnostics']:
                layer = diag.layer
                d_dict = diag.to_dict()
                prefix = f"L{layer}_"
                row[f"{prefix}energy"] = d_dict['energy']
                row[f"{prefix}smoothness"] = d_dict['smoothness_index']
                row[f"{prefix}entropy"] = d_dict['spectral_entropy']
                row[f"{prefix}hfer"] = d_dict['hfer']
                row[f"{prefix}fiedler"] = d_dict['fiedler_value']
                row[f"{prefix}radius"] = d_dict.get('spectral_radius')
                row[f"{prefix}max_imag"] = d_dict.get('max_imaginary')
                row[f"{prefix}gini"] = d_dict.get('gini_sparsity')
                row[f"{prefix}attn_gini"] = d_dict.get('attention_gini')
                
                # Spectral Velocity (Fiedler derivative)
                if layer > 0:
                    prev_fiedler = row[f"L{layer-1}_fiedler"]
                    row[f"{prefix}velocity"] = d_dict['fiedler_value'] - prev_fiedler
                else:
                    row[f"{prefix}velocity"] = 0.0

            results_rows.append(row)

    df = pd.DataFrame(results_rows)
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/riabi_features_N200.csv', index=False)
    print(f"Extraction complete. Saved to data/riabi_features_N200.csv")

if __name__ == "__main__":
    extract()
