import os
import json
import torch
import numpy as np
import random
import subprocess
import tempfile
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from difflib import SequenceMatcher

# --- UTILS ---

def compute_hfer_single_layer(attention_matrix):
    """Computes HFER (Fiedler / Max Eigenvalue) for a single attention slice with symmetrization."""
    A = attention_matrix.mean(dim=0).float().cpu().numpy()
    # Symmetrize attention matrix (Standard procedure in Spectral Graph Theory for directed graphs)
    A_sym = 0.5 * (A + A.T)
    # Row sums for Degree matrix (A_sym is symmetric, so row sums = column sums)
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
    """Returns HFER at layer 21 for the given text."""
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)
        # Layer 21 is index 21
        h21 = compute_hfer_single_layer(outputs.attentions[21][0])
        del outputs
        torch.cuda.empty_cache()
    return h21

def verify_lean_proof(proof_text, theorem_statement):
    """Writes a temporary .lean file and runs Lean 4 to verify."""
    # Combine theorem statement and proof
    # Most proofs in MiniF2F start with 'proof' or 'by'
    # We'll try to reconstruct a valid Lean file
    content = f"{theorem_statement}\n{proof_text}"
    
    with tempfile.NamedTemporaryFile(suffix=".lean", delete=False, mode='w', encoding='utf-8') as f:
        f.write(content)
        temp_path = f.name
    
    try:
        # Run lean command
        # Use subprocess.run with a timeout
        result = subprocess.run(["lean", temp_path], capture_output=True, text=True, timeout=30)
        is_valid = (result.returncode == 0)
    except Exception as e:
        print(f"Lean error: {e}")
        is_valid = False
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
            
    return is_valid

# --- MAIN EXPERIMENT ---

def run_best_of_n():
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
    import re
    theorems = []
    data_dirs = ['data/experiment_ready/valid', 'data/experiment_ready/invalid']
    # Split pattern for various Lean styles
    split_pattern = re.compile(r'\b(by|proof|begin|:=)\b')
    for d in data_dirs:
        if not os.path.exists(d): continue
        for f in os.listdir(d):
            if f.endswith('.lean'):
                path = os.path.join(d, f)
                with open(path, 'r', encoding='utf-8') as file:
                    content = file.read()
                    # Extract statement
                    match = split_pattern.search(content)
                    if match:
                        statement = content[:match.start()].strip()
                        theorems.append({'name': f, 'statement': statement, 'full_content': content})

    print(f"Total theorems loaded: {len(theorems)}")

    # 2. Baseline Greedy Pass
    print("Running baseline greedy pass...")
    baseline_results = []
    for th in tqdm(theorems, desc="Greedy"):
        inputs = tokenizer(th['statement'], return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=256, do_sample=False, return_dict_in_generate=True, output_attentions=True)
            gen_text = tokenizer.decode(out.sequences[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            # HFER at L21 for the generated sequence
            h21 = get_hfer_l21(model, tokenizer, th['statement'] + gen_text)
            baseline_results.append({'name': th['name'], 'statement': th['statement'], 'h21': h21, 'proof': gen_text})

    # Pick 40 middle-difficulty theorems (50th to 80th percentile)
    # This avoids the "impossible" ones and the "trivial" ones,
    # targeting the regime where Best-of-N filtering is most relevant.
    # Pick 40 theorems with highest HFER (likely invalid/difficult)
    worst_40 = sorted(baseline_results, key=lambda x: x['h21'], reverse=True)[:40]
    print(f"Selected 40 theorems with highest HFER. Range: {worst_40[-1]['h21']:.4f} to {worst_40[0]['h21']:.4f}")

    # 3. Sampling 8 candidates for each
    print("Generating 8 sampled candidates with formal prompting...")
    final_results = []
    
    # Prompt template for Instruct model
    sys_msg = "You are a Lean 4 formal prover. Generate only the proof block starting with 'by'. Do not explain."
    
    for th in tqdm(worst_40, desc="Sampling"):
        candidates = []
        # Construct Instruct prompt
        full_prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{sys_msg}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{th['statement']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nby"
        
        inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)
        
        for i in range(8):
            with torch.no_grad():
                out = model.generate(
                    **inputs, 
                    max_new_tokens=256, 
                    do_sample=True, 
                    temperature=0.7, 
                    top_p=0.95, 
                    return_dict_in_generate=True, 
                    output_scores=True
                )
                # Decode from the end of the prompt (the model continues from 'by')
                gen_text = "by" + tokenizer.decode(out.sequences[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
                
                # Metrics
                # Transition scores for logprobs
                t_scores = model.compute_transition_scores(out.sequences, out.scores, normalize_logits=True)
                mean_lp = float(np.mean(t_scores[0].cpu().numpy()))
                
                h21 = get_hfer_l21(model, tokenizer, th['statement'] + gen_text)
                
                candidates.append({
                    "id": i,
                    "text": gen_text,
                    "h21": h21,
                    "mean_log_prob": mean_lp,
                    "length": len(gen_text)
                })
        
        # Pre-verify all candidates for this theorem once
        for cand in candidates:
            cand['is_valid'] = verify_lean_proof(cand['text'], th['statement'])
        
        # Apply Selection Strategies
        # N=4 and N=8
        strategies = {}
        for N in [4, 8]:
            subset = candidates[:N]
            
            # Random selection (average over 50 repeats)
            random_validity = []
            for _ in range(50):
                chosen = random.choice(subset)
                random_validity.append(chosen['is_valid'])
            
            # Highest Log-prob
            best_lp = max(subset, key=lambda x: x['mean_log_prob'])
            valid_lp = best_lp['is_valid']
            
            # Lowest HFER
            best_hfer = min(subset, key=lambda x: x['h21'])
            valid_hfer = best_hfer['is_valid']
            
            strategies[f"N={N}"] = {
                "random_valid_rate": np.mean(random_validity),
                "max_lp": {"h21": best_lp['h21'], "valid": valid_lp},
                "min_hfer": {"h21": best_hfer['h21'], "valid": valid_hfer}
            }
            
        final_results.append({
            "theorem_name": th['name'],
            "theorem_statement": th['statement'],
            "candidates": candidates,
            "strategies": strategies
        })

    # Save everything
    os.makedirs("data/results/rebuttal", exist_ok=True)
    with open("data/results/rebuttal/best_of_n_final.json", "w") as f:
        json.dump(final_results, f, indent=2)

    # Print Summary Report
    print("\n" + "="*50)
    print("  EXPERIMENT 1 SUMMARY REPORT")
    print("="*50)
    for N in [4, 8]:
        results_N = [r['strategies'][f"N={N}"] for r in final_results]
        
        rand_rate = np.mean([r['random_valid_rate'] for r in results_N])
        lp_rate = np.mean([1 if r['max_lp']['valid'] else 0 for r in results_N])
        hfer_rate = np.mean([1 if r['min_hfer']['valid'] else 0 for r in results_N])
        
        lp_hfer_val = np.mean([r['max_lp']['h21'] for r in results_N])
        hfer_hfer_val = np.mean([r['min_hfer']['h21'] for r in results_N])
        
        print(f"\n[N={N}]")
        print(f"  Random validity rate:    {rand_rate:.2%}")
        print(f"  Max Log-prob validity:   {lp_rate:.2%}")
        print(f"  Min HFER validity:       {hfer_rate:.2%}")
        print(f"  Mean HFER (Max LP):      {lp_hfer_val:.4f}")
        print(f"  Mean HFER (Min HFER):    {hfer_hfer_val:.4f}")

if __name__ == "__main__":
    run_best_of_n()
