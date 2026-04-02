import os
import json
import torch
import numpy as np
import random
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from difflib import SequenceMatcher

# --- UTILS ---

def compute_hfer_single_layer(attention_matrix):
    A = attention_matrix.mean(dim=0).float().cpu().numpy()
    A_sym = 0.5 * (A + A.T)
    D = np.diag(np.sum(A_sym, axis=1))
    L = D - A_sym
    try:
        vals = np.linalg.eigvalsh(L)
        vals = np.sort(vals)
        fiedler = vals[1] if len(vals) > 1 else 0
        return float(fiedler / (vals[-1] + 1e-6))
    except Exception:
        return 0.0

def get_hfer_l21(model, tokenizer, text):
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)
        h21 = compute_hfer_single_layer(outputs.attentions[21][0])
        del outputs
        torch.cuda.empty_cache()
    return h21

def apply_spectral_steering(model, layer_idx, alpha=-0.3):
    """Apply spectral steering to mlp.down_proj at given layer."""
    device = model.model.layers[layer_idx].mlp.down_proj.weight.device
    dtype = model.model.layers[layer_idx].mlp.down_proj.weight.dtype
    
    # We move to CPU because SVD is more stable and sometimes faster there for medium-sized matrices
    W = model.model.layers[layer_idx].mlp.down_proj.weight.data.float().cpu()
    U, S, Vh = torch.linalg.svd(W, full_matrices=False)
    
    # S_new = S * (1 + alpha * (S - S.mean()) / S.std())
    # Note: Using torch operations for stability
    S_mean = torch.mean(S)
    S_std = torch.std(S)
    S_new = S * (1 + alpha * (S - S_mean) / S_std)
    
    W_new = U @ torch.diag(S_new) @ Vh
    model.model.layers[layer_idx].mlp.down_proj.weight.data = W_new.to(device=device, dtype=dtype)
    return model

# --- MAIN EXPERIMENT ---

def run_steering_behavioral():
    model_id = "meta-llama/Llama-3.2-3B-Instruct"
    print(f"Loading {model_id} in 4-bit...")
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        output_attentions=True
    )

    # 1. Load Theorems
    print("Loading theorems...")
    theorems = []
    data_dirs = ['data/experiment_ready/valid', 'data/experiment_ready/invalid']
    for d in data_dirs:
        for f in os.listdir(d):
            if f.endswith('.lean'):
                path = os.path.join(d, f)
                with open(path, 'r', encoding='utf-8') as file:
                    content = file.read()
                    parts = content.split('proof')
                    if len(parts) < 2: parts = content.split('by')
                    if len(parts) >= 2:
                        theorems.append({'name': f, 'statement': parts[0].strip(), 'full_content': content})

    print(f"Total theorems loaded: {len(theorems)}")

    # 2. Pick 50 theorems (25 high HFER, 25 low HFER from base)
    print("Running initial pass to pick 50 theorems...")
    baseline_results = []
    for th in tqdm(theorems, desc="Initial Pass"):
        inputs = tokenizer(th['statement'], return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=256, do_sample=False, return_dict_in_generate=True, output_attentions=True)
            gen_text = tokenizer.decode(out.sequences[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            h21 = get_hfer_l21(model, tokenizer, th['statement'] + gen_text)
            baseline_results.append({'name': th['name'], 'statement': th['statement'], 'h21': h21, 'proof': gen_text})

    high_hfer = sorted(baseline_results, key=lambda x: x['h21'], reverse=True)[:25]
    low_hfer = sorted(baseline_results, key=lambda x: x['h21'])[:25]
    selected_theorems = high_hfer + low_hfer
    print(f"Selected 25 high HFER ({high_hfer[-1]['h21']:.4f} to {high_hfer[0]['h21']:.4f})")
    print(f"Selected 25 low HFER ({low_hfer[0]['h21']:.4f} to {low_hfer[-1]['h21']:.4f})")

    # 3. Apply Steering
    print("\nApplying Spectral Steering (alpha=-0.3 at Layer 21)...")
    apply_spectral_steering(model, 21, alpha=-0.3)

    # 4. Generate Steered Proofs
    print("Generating steered proofs...")
    final_results = []
    for th in tqdm(selected_theorems, desc="Steered Pass"):
        inputs = tokenizer(th['statement'], return_tensors="pt").to(model.device)
        with torch.no_grad():
            # Same parameters (greedy)
            out = model.generate(**inputs, max_new_tokens=256, do_sample=False, return_dict_in_generate=True, output_attentions=True)
            steered_proof = tokenizer.decode(out.sequences[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            h21_steered = get_hfer_l21(model, tokenizer, th['statement'] + steered_proof)
            
            # Compare texts
            base_proof = th['proof']
            identical = (base_proof.strip() == steered_proof.strip())
            ratio = SequenceMatcher(None, base_proof.strip(), steered_proof.strip()).ratio()
            
            final_results.append({
                "theorem_name": th['name'],
                "base_proof": base_proof,
                "steered_proof": steered_proof,
                "base_hfer": th['h21'],
                "steered_hfer": h21_steered,
                "text_identical": identical,
                "similarity_ratio": ratio
            })

    # Save results
    os.makedirs("data/results/rebuttal", exist_ok=True)
    with open("data/results/rebuttal/steering_behavioral_final.json", "w") as f:
        json.dump(final_results, f, indent=2)

    # Print Summary Report
    print("\n" + "="*50)
    print("  EXPERIMENT 2 SUMMARY REPORT")
    print("="*50)
    
    num_changed = sum(1 for r in final_results if not r['text_identical'])
    avg_ratio_changed = np.mean([r['similarity_ratio'] for r in final_results if not r['text_identical']]) if num_changed > 0 else 1.0
    
    h_subset = final_results[:25]
    l_subset = final_results[25:]
    
    def get_hfer_change(subset):
        base = np.mean([r['base_hfer'] for r in subset])
        steer = np.mean([r['steered_hfer'] for r in subset])
        return base, steer
        
    h_base, h_steer = get_hfer_change(h_subset)
    l_base, l_steer = get_hfer_change(l_subset)
    
    print(f"\n[Overall]")
    print(f"  Proofs changed text:      {num_changed}/50 ({num_changed/50:.1%})")
    print(f"  Mean similarity ratio:    {avg_ratio_changed:.4f} (for changed proofs)")
    
    print(f"\n[High HFER Subset (Invalid-ish)]")
    print(f"  Mean HFER (Base):         {h_base:.4f}")
    print(f"  Mean HFER (Steered):      {h_steer:.4f}")
    print(f"  Delta:                    {h_steer - h_base:+.4f}")
    
    print(f"\n[Low HFER Subset (Valid-ish)]")
    print(f"  Mean HFER (Base):         {l_base:.4f}")
    print(f"  Mean HFER (Steered):      {l_steer:.4f}")
    print(f"  Delta:                    {l_steer - l_base:+.4f}")
    
    print("\nFirst 5 pairs (Base vs Steered):")
    for i in range(5):
        r = final_results[i]
        print(f"\n--- Theorem: {r['theorem_name']} ---")
        print(f"  Base Proof:    {r['base_proof'][:100]}...")
        print(f"  Steered Proof: {r['steered_proof'][:100]}...")
        print(f"  Identical?     {r['text_identical']}")

if __name__ == "__main__":
    run_steering_behavioral()
