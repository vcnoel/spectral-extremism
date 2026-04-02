import os
import json
import subprocess
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

def verify_lean4_proof(proof_text, theorem_statement):
    """Verifies a Lean 4 proof block against a theorem statement."""
    content = f"{theorem_statement}\n{proof_text}"
    with open("tmp_pass1_check.lean", "w", encoding="utf-8") as f:
        f.write(content)
    try:
        result = subprocess.run(['lean', 'tmp_pass1_check.lean'], capture_output=True, text=True, timeout=10)
        return result.returncode == 0
    except:
        return False

def run_pass1_baseline():
    model_id = "meta-llama/Llama-3.1-8B-Instruct"
    print(f"Loading {model_id} for Pass@1 Baseline...")
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map="auto")

    # The 13 verified Lean 4 MiniF2F equivalents
    theorems = [
        {'name': 'mathd_algebra_359.lean', 'statement': 'import Mathlib.Tactic\ntheorem mathd_algebra_359 (y : \u211d) : (y + 6) * (y + 6) = y^2 + 12 * y + 36 := '},
        {'name': 'mathd_numbertheory_66.lean', 'statement': 'import Mathlib.Tactic\ntheorem mathd_numbertheory_66 : 194 % 11 = 7 := '},
        {'name': 'mathd_algebra_140.lean', 'statement': 'import Mathlib.Tactic\ntheorem mathd_algebra_140 (x : \u211d) (h0 : x = 2 * 24) : x = 48 := '},
        {'name': 'mathd_algebra_15.lean', 'statement': 'import Mathlib.Tactic\ntheorem mathd_algebra_15 (s : \u2115) (h0 : s = 2^1 + 2^2 + 2^3) : s = 14 := '},
        {'name': 'mathd_numbertheory_3.lean', 'statement': 'import Mathlib.Tactic\ntheorem mathd_numbertheory_3 : (12 * 7) % 5 = 4 := '},
        {'name': 'amc12b_2020_p2.lean', 'statement': 'import Mathlib.Tactic\ntheorem amc12b_2020_p2 (x : \u211d) (h0 : (x - 20)^2 = 100) : (x-10)*(x-30) = 0 := '},
        {'name': 'mathd_algebra_10.lean', 'statement': 'import Mathlib.Tactic\ntheorem mathd_algebra_10 (x : \u211d) (h0 : x = 12 / 3) : x = 4 := '},
        {'name': 'mathd_numbertheory_342.lean', 'statement': 'import Mathlib.Tactic\ntheorem mathd_numbertheory_342 : 5^4 % 6 = 1 := '},
        {'name': 'mathd_algebra_182.lean', 'statement': 'import Mathlib.Tactic\ntheorem mathd_algebra_182 (y : \u211d) : 7 * (2 * y + 3) = 14 * y + 21 := '},
        {'name': 'mathd_algebra_24.lean', 'statement': 'import Mathlib.Tactic\ntheorem mathd_algebra_24 (x : \u211d) (h0 : x = 100) : x / 10 = 10 := '},
        {'name': 'mathd_numbertheory_101.lean', 'statement': 'import Mathlib.Tactic\ntheorem mathd_numbertheory_101 : (17 * 18 * 19 * 20) % 10 = 0 := '},
        {'name': 'mathd_algebra_141.lean', 'statement': 'import Mathlib.Tactic\ntheorem mathd_algebra_141 (a b : \u211d) (h0 : a^2 + b^2 = 0) : a = 0 \u2227 b = 0 := '},
        {'name': 'mathd_algebra_139.lean', 'statement': 'import Mathlib.Tactic\ntheorem mathd_algebra_139 (s : \u211d) (h0 : s / 5 + 4 = 0) : s = -20 := '}
    ]

    results = []
    print("\n--- GROUND-TRUTH PASS@1 BASELINE (8B) ---")
    correct_count = 0
    
    for th in tqdm(theorems):
        with torch.no_grad():
            inputs = tokenizer(th['statement'], return_tensors="pt").to(model.device)
            out = model.generate(**inputs, max_new_tokens=100, do_sample=False)
            gen_text = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            
            is_valid = verify_lean4_proof(gen_text, th['statement'])
            if is_valid: correct_count += 1
            results.append({"name": th['name'], "is_valid": is_valid})
            
    print(f"\nFinal Ground-Truth Pass@1 (Greedy): {(correct_count/len(theorems)):.2%}")
    with open("data/results/rebuttal/verify_lean4_pass1.json", "w") as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    run_pass1_baseline()
