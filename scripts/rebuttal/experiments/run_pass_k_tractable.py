import os
import json
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import subprocess

# --- UTILS ---

def verify_lean_proof(proof_text, theorem_statement):
    """Verifies a Lean proof block against a theorem statement."""
    content = f"import Mathlib.Tactic\n{theorem_statement}\n{proof_text}"
    with open("tmp_pass_k_check.lean", "w", encoding="utf-8") as f:
        f.write(content)
    try:
        # Lean 4 verification
        result = subprocess.run(['lean', 'tmp_pass_k_check.lean'], capture_output=True, text=True, timeout=15)
        return result.returncode == 0
    except:
        return False

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
    except: return 1.0

def get_hfer_l21(model, tokenizer, text):
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)
        h21 = compute_hfer_single_layer(outputs.attentions[21][0])
        del outputs
        torch.cuda.empty_cache()
    return h21

# --- EXPERIMENT ---

def run_experiment():
    model_id = "meta-llama/Llama-3.1-8B-Instruct"
    print(f"Loading {model_id} for Pass@k filtering experiment...")
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map="auto", output_attentions=True)

    # 20 Tractable MiniF2F-style theorems (Lean 4)
    theorems = [
        {"name": "mathd_algebra_359", "statement": "theorem mathd_algebra_359 (y : \u211d) : (y + 6) * (y + 6) = y^2 + 12 * y + 36 := "},
        {"name": "mathd_numbertheory_66", "statement": "theorem mathd_numbertheory_66 : 194 % 11 = 7 := "},
        {"name": "mathd_algebra_140", "statement": "theorem mathd_algebra_140 (x : \u211d) (h0 : x = 2 * 24) : x = 48 := "},
        {"name": "mathd_algebra_15", "statement": "theorem mathd_algebra_15 (s : \u2115) (h0 : s = 2^1 + 2^2 + 2^3) : s = 14 := "},
        {"name": "mathd_numbertheory_3", "statement": "theorem mathd_numbertheory_3 : (12 * 7) % 5 = 4 := "},
        {"name": "mathd_algebra_10", "statement": "theorem mathd_algebra_10 (x : \u211d) (h0 : x = 12 / 3) : x = 4 := "},
        {"name": "mathd_numbertheory_342", "statement": "theorem mathd_numbertheory_342 : 5^4 % 6 = 1 := "},
        {"name": "mathd_algebra_182", "statement": "theorem mathd_algebra_182 (y : \u211d) : 7 * (2 * y + 3) = 14 * y + 21 := "},
        {"name": "mathd_algebra_24", "statement": "theorem mathd_algebra_24 (x : \u211d) (h0 : x = 100) : x / 10 = 10 := "},
        {"name": "mathd_numbertheory_101", "statement": "theorem mathd_numbertheory_101 : (17 * 18 * 19 * 20) % 10 = 0 := "},
        # Adding more basic ones
        {"name": "mathd_algebra_141", "statement": "theorem mathd_algebra_141 (a b : \u211d) (h0 : a^2+b^2=0) : a=0 \u2227 b=0 := "},
        {"name": "mathd_algebra_13", "statement": "theorem mathd_algebra_13 (a : \u211d) : (a + 1)^2 - (a - 1)^2 = 4 * a := "},
        {"name": "mathd_numbertheory_1", "statement": "theorem mathd_numbertheory_1 : 12345 % 2 = 1 := "},
        {"name": "mathd_algebra_2", "statement": "theorem mathd_algebra_2 (x : \u211d) (h0 : 3*x + 1 = 10) : x = 3 := "},
        {"name": "mathd_algebra_3", "statement": "theorem mathd_algebra_3 (a : \u211d) (h0 : a = 5) : a^2 = 25 := "},
        {"name": "mathd_numbertheory_4", "statement": "theorem mathd_numbertheory_4 : (4 * 3) % 7 = 5 := "},
        {"name": "mathd_algebra_5", "statement": "theorem mathd_algebra_5 (x : \u211d) (h0 : x / 2 = 10) : x = 20 := "},
        {"name": "mathd_algebra_6", "statement": "theorem mathd_algebra_6 (y : \u211d) (h0 : y^2 = 49 \u2227 y > 0) : y = 7 := "},
        {"name": "mathd_numbertheory_7", "statement": "theorem mathd_numbertheory_7 : (2^3) % 5 = 3 := "},
        {"name": "mathd_algebra_8", "statement": "theorem mathd_algebra_8 (x : \u211d) (h0 : x + x = 10) : x = 5 := "}
    ]

    results = []
    print("\n--- EXPERIMENT 1: PASS@K (TRACTABLE SUBSET) ---")
    
    for th in tqdm(theorems):
        candidates = []
        prompt = th['statement']
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        for i in range(8):
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=64, do_sample=True, temperature=0.7, return_dict_in_generate=True, output_scores=True)
                gen_text = tokenizer.decode(out.sequences[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
                
                is_valid = verify_lean_proof(gen_text, th['statement'])
                hfer = get_hfer_l21(model, tokenizer, prompt + gen_text)
                
                t_scores = model.compute_transition_scores(out.sequences, out.scores, normalize_logits=True)
                mean_lp = float(np.mean(t_scores[0].cpu().numpy()))
                
                candidates.append({"id": i, "is_valid": is_valid, "hfer": hfer, "mean_lp": mean_lp})
        
        results.append({"name": th['name'], "candidates": candidates})
        # Save incrementally
        with open("data/results/rebuttal/pass_k_tractable.json", "w") as f:
            json.dump(results, f, indent=2)

    # FINAL METRICS
    print("\n" + "="*50)
    print("  PASS@K FILTERING SUMMARY")
    print("="*50)
    
    # Calculate Random Pass@1
    total_valid = sum(sum(1 if c['is_valid'] else 0 for c in p['candidates']) for p in results)
    total_candidates = len(results) * 8
    random_p1 = total_valid / total_candidates
    
    # Calculate HFER Selector Pass@1
    hfer_p1 = sum(1 if min(p['candidates'], key=lambda x: x['hfer'])['is_valid'] else 0 for p in results) / len(results)
    
    # Calculate LogProb Selector Pass@1
    lp_p1 = sum(1 if max(p['candidates'], key=lambda x: x['mean_lp'])['is_valid'] else 0 for p in results) / len(results)
    
    print(f"Pass@1 (Random):      {random_p1:.2%}")
    print(f"Pass@1 (Max LogProb): {lp_p1:.2%}")
    print(f"Pass@1 (Min HFER):    {hfer_p1:.2%}")
    print("="*50)

if __name__ == "__main__":
    run_experiment()
