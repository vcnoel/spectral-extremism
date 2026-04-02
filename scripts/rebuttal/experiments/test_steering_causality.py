import os
import json
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import subprocess

# --- STEERING UTILS ---

def apply_steering_hook(model, layer_idx, alpha=-0.3):
    """Applies a spectral steering hook to the MLP output."""
    def hook(module, input, output):
        # We simulate the weight edit shift in activation space
        return output * (1.0 + alpha)
    
    handle = model.model.layers[layer_idx].mlp.register_forward_hook(hook)
    return handle

def verify_lean_proof(proof_text, theorem_statement):
    content = f"import Mathlib.Tactic\n{theorem_statement}\n{proof_text}"
    with open("tmp_steering_check.lean", "w", encoding="utf-8") as f:
        f.write(content)
    try:
        result = subprocess.run(['lean', 'tmp_steering_check.lean'], capture_output=True, text=True, timeout=15)
        return result.returncode == 0
    except:
        return False

# --- PILOT STUDY ---

def run_steering_pilot():
    model_id = "meta-llama/Llama-3.1-8B-Instruct"
    print(f"Loading {model_id} for Steering Pilot...")
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map="auto")

    # Near-miss proofs (previously identified)
    near_miss_theorems = [
        {"name": "mathd_algebra_10_3", "statement": "theorem mathd_algebra_10 (x : ℝ) (h0 : x = 12 / 3) : x = 4 := "},
        {"name": "mathd_algebra_15_1", "statement": "theorem mathd_algebra_15 (s : ℕ) (h0 : s = 2^1 + 2^2 + 2^3) : s = 14 := "},
        {"name": "mathd_algebra_109_2", "statement": "theorem mathd_algebra_109 (a b : ℝ) (h0 : 3*a + 2*b = 12) (h1 : a = 4) : b = 0 := "},
        {"name": "mathd_algebra_123_3", "statement": "theorem mathd_algebra_123 (a b : ℕ) (h0 : a + b = 20) (h1 : a = 5) : b = 15 := "},
        {"name": "mathd_algebra_51_5", "statement": "theorem mathd_algebra_51 (x y : ℝ) (h0 : x + y = 10) (h1 : x - y = 2) : x = 6 := "},
        {"name": "induction_div_9_10nm1_3", "statement": "theorem induction_div_9_10nm1 (n : ℕ) : 9 ∣ (10^n - 1) := "}
    ]

    # Baseline (Before steering)
    print("\n--- BASELINE GENERATION ---")
    baseline_success = 0
    for th in near_miss_theorems:
        with torch.no_grad():
            inputs = tokenizer(th['statement'], return_tensors="pt").to(model.device)
            out = model.generate(**inputs, max_new_tokens=48, do_sample=False)
            gen_text = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            valid = verify_lean_proof(gen_text, th['statement'])
            if valid: baseline_success += 1
            print(f"[{th['name']}] Baseline Valid: {valid}")

    # Apply Steering via Hook
    print("\nApplying spectral steering hook (alpha=-0.3) to Layer 21")
    handle = apply_steering_hook(model, 21, alpha=-0.3)
    
    print("\n--- STEERED GENERATION ---")
    steered_success = 0
    for th in near_miss_theorems:
        with torch.no_grad():
            inputs = tokenizer(th['statement'], return_tensors="pt").to(model.device)
            out = model.generate(**inputs, max_new_tokens=48, do_sample=False)
            gen_text = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
            valid = verify_lean_proof(gen_text, th['statement'])
            if valid: steered_success += 1
            print(f"[{th['name']}] Steered Valid: {valid}")
    
    handle.remove()

    print("\n" + "="*50)
    print("  STEERING CAUSAL SHIFT SUMMARY")
    print("="*50)
    print(f"  Baseline Success: {baseline_success}/{len(near_miss_theorems)}")
    print(f"  Steered Success:  {steered_success}/{len(near_miss_theorems)}")
    print(f"  Success Rate Delta: {(steered_success - baseline_success)/len(near_miss_theorems):+.2%}")
    print("="*50)

if __name__ == "__main__":
    run_steering_pilot()
