import os
import json
import torch
import numpy as np
import random
import subprocess
import tempfile
import re
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from scipy.special import comb

# --- UTILS ---

def compute_hfer_single_layer(attention_matrix):
    """Computes HFER (Fiedler / Max Eigenvalue) for single attention matrix with symmetrization."""
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
    """Extracts HFER at layer 21 for the provided text."""
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)
        # Llama-3.1-8B Layer 21 HFER (index 21)
        h21 = compute_hfer_single_layer(outputs.attentions[21][0])
        del outputs
        torch.cuda.empty_cache()
    return h21

def verify_lean_proof(proof_text, theorem_statement):
    """Verifies a Lean 4 proof block against a theorem statement."""
    content = f"{theorem_statement}\n{proof_text}"
    with tempfile.NamedTemporaryFile(suffix=".lean", delete=False, mode='w', encoding='utf-8') as f:
        f.write(content)
        temp_path = f.name
    try:
        result = subprocess.run(["lean", temp_path], capture_output=True, text=True, timeout=30)
        is_valid = (result.returncode == 0)
    except Exception:
        is_valid = False
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
    return is_valid

def calculate_pass_at_k(n, c, k):
    """Computes Pass@k unbiased estimator: 1 - comb(n-c, k) / comb(n, k)."""
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)

# --- MAIN EXPERIMENT ---

def run_experiment():
    model_id = "meta-llama/Llama-3.1-8B-Instruct"
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

    # 1. Load Theorems (Core Lean 4 - No imports required for speed and reliability)
    print("Loading Core Lean 4 theorem subset (13 theorems)...")
    theorems = [
        {'name': 'core_add_1.lean', 'statement': 'theorem core_add_1 : 2 + 2 = 4 := '},
        {'name': 'core_mul_1.lean', 'statement': 'theorem core_mul_1 : 5 * 6 = 30 := '},
        {'name': 'core_logic_1.lean', 'statement': 'theorem core_logic_1 (p q : Prop) : p \u2227 q \u2192 p := '},
        {'name': 'core_logic_2.lean', 'statement': 'theorem core_logic_2 (p q : Prop) : p \u2192 p \u2228 q := '},
        {'name': 'core_nat_1.lean', 'statement': 'theorem core_nat_1 (n : Nat) : n + 0 = n := '},
        {'name': 'core_sub_1.lean', 'statement': 'theorem core_sub_1 : 100 - 50 = 50 := '},
        {'name': 'core_succ_1.lean', 'statement': 'theorem core_succ_1 : Nat.succ 9 = 10 := '},
        {'name': 'core_refl_1.lean', 'statement': 'theorem core_refl_1 (n : Nat) : n = n := '},
        {'name': 'core_list_1.lean', 'statement': 'theorem core_list_1 : List.length [1, 2, 3] = 3 := '},
        {'name': 'core_logic_3.lean', 'statement': 'theorem core_logic_3 (p : Prop) : \u00ac(p \u2227 \u00acp) := '},
        {'name': 'core_exp_1.lean', 'statement': 'theorem core_exp_1 : 2^3 = 8 := '},
        {'name': 'core_mod_1.lean', 'statement': 'theorem core_mod_1 : 15 % 4 = 3 := '},
        {'name': 'core_mul_2.lean', 'statement': 'theorem core_mul_2 : 10 * 10 = 100 := '}
    ]
    print(f"Loaded {len(theorems)} Core Lean 4 theorems.")

    # 2. Generate 8 candidates and verify
    sys_msg = "You are a Lean 4 formal prover. Generate only the proof block starting with 'by'. Do not explain."
    results = []
    output_path = "data/results/rebuttal/best_of_n_8b.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    total_theorems = len(theorems)
    for idx, th in enumerate(theorems):
        print(f"[{idx+1}/{total_theorems}] Processing: {th['name']}")
        candidates = []
        prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{sys_msg}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{th['statement']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nby"
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        for i in range(8):
            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=True,
                    temperature=0.8,
                    top_p=0.95,
                    return_dict_in_generate=True,
                    output_scores=True
                )
                gen_text = "by" + tokenizer.decode(out.sequences[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
                
                # Metrics
                t_scores = model.compute_transition_scores(out.sequences, out.scores, normalize_logits=True)
                mean_lp = float(np.mean(t_scores[0].cpu().numpy()))
                h21 = get_hfer_l21(model, tokenizer, th['statement'] + gen_text)
                is_valid = verify_lean_proof(gen_text, th['statement'])
                
                candidates.append({
                    "id": i,
                    "text": gen_text,
                    "h21": h21,
                    "mean_log_prob": mean_lp,
                    "is_valid": is_valid
                })

        # Calculate Pass@k and Best-of-N metrics for this theorem
        correct_count = sum(1 for c in candidates if c['is_valid'])
        
        # Pass@k
        pass_at_1 = calculate_pass_at_k(8, correct_count, 1)
        pass_at_4 = calculate_pass_at_k(8, correct_count, 4)
        pass_at_8 = calculate_pass_at_k(8, correct_count, 8)
        
        # Best-of-N (Selectors: HFER vs Logprob)
        # N=4
        sub_4 = candidates[:4]
        bon_lp_4 = max(sub_4, key=lambda x: x['mean_log_prob'])['is_valid']
        bon_hfer_4 = min(sub_4, key=lambda x: x['h21'])['is_valid']
        
        # N=8
        bon_lp_8 = max(candidates, key=lambda x: x['mean_log_prob'])['is_valid']
        bon_hfer_8 = min(candidates, key=lambda x: x['h21'])['is_valid']
        
        results.append({
            "name": th['name'],
            "correct_count": correct_count,
            "pass_at_1": pass_at_1,
            "pass_at_4": pass_at_4,
            "pass_at_8": pass_at_8,
            "bon_lp_4": bon_lp_4,
            "bon_hfer_4": bon_hfer_4,
            "bon_lp_8": bon_lp_8,
            "bon_hfer_8": bon_hfer_8
        })
        
        # Incremental save
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

    # Summary Statistics
    summary = {
        "pass_at_1": np.mean([r['pass_at_1'] for r in results]),
        "pass_at_4": np.mean([r['pass_at_4'] for r in results]),
        "pass_at_8": np.mean([r['pass_at_8'] for r in results]),
        "bon_lp_4": np.mean([r['bon_lp_4'] for r in results]),
        "bon_hfer_4": np.mean([r['bon_hfer_4'] for r in results]),
        "bon_lp_8": np.mean([r['bon_lp_8'] for r in results]),
        "bon_hfer_8": np.mean([r['bon_hfer_8'] for r in results])
    }

    print("\n" + "="*50)
    print(f"  EXPERIMENT 1 SUMMARY REPORT (LLAMA-3.1-8B)")
    print("="*50)
    print(f"Pass@1:     {summary['pass_at_1']:.2%}")
    print(f"Pass@4:     {summary['pass_at_4']:.2%}")
    print(f"Pass@8:     {summary['pass_at_8']:.2%}")
    print("-" * 20)
    print(f"Best-of-4 (Logprob): {summary['bon_lp_4']:.2%}")
    print(f"Best-of-4 (HFER):    {summary['bon_hfer_4']:.2%}")
    print(f"Best-of-8 (Logprob): {summary['bon_lp_8']:.2%}")
    print(f"Best-of-8 (HFER):    {summary['bon_hfer_8']:.2%}")
    print("="*50)

if __name__ == "__main__":
    run_experiment()
