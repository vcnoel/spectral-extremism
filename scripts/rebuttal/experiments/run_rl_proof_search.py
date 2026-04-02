import os
import json
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# --- SPECTRAL RL UTILS ---

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

def get_hfer_reward(model, tokenizer, text):
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)
        h21 = compute_hfer_single_layer(outputs.attentions[21][0])
        del outputs
        torch.cuda.empty_cache()
    return -h21 # Maximizing reward = minimizing HFER

# --- EXPERIMENT ---

def run_rl_search():
    model_id = "meta-llama/Llama-3.1-8B-Instruct"
    print(f"Loading {model_id} for RL Search Study...")
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map="auto", output_attentions=True)

    # Use the 13 verified Lean 4 MiniF2F equivalents
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
    print("\n--- SPECTRAL RL SEARCH (N=4 Branches, Depth=3) ---")
    
    for th in theorems:
        print(f"\nProblem: {th['name']}")
        current_proof = "by "
        
        # RL Search for 3 steps
        for step in range(3):
            prompt = th['statement'] + current_proof
            candidates = []
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            # Generate 4 branches
            for _ in range(4):
                with torch.no_grad():
                    out = model.generate(**inputs, max_new_tokens=32, do_sample=True, temperature=0.7, return_dict_in_generate=True, output_scores=True)
                    gen_text = tokenizer.decode(out.sequences[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).split('\n')[0]
                    
                    t_scores = model.compute_transition_scores(out.sequences, out.scores, normalize_logits=True)
                    mean_lp = float(np.mean(t_scores[0].cpu().numpy()))
                    
                    reward = get_hfer_reward(model, tokenizer, prompt + gen_text)
                    candidates.append({"text": gen_text, "reward": reward, "lp": mean_lp})
            
            # RL Selection (minimize HFER)
            best_idx = np.argmax([c['reward'] for c in candidates])
            greedy_idx = np.argmax([c['lp'] for c in candidates])
            
            print(f"Step {step}: Selected '{candidates[best_idx]['text']}' (HFER reward {candidates[best_idx]['reward']:.3f}) vs greedy '{candidates[greedy_idx]['text']}'")
            current_proof += candidates[best_idx]['text'] + " "
            
        results.append({"name": th['name'], "final_proof": th['statement'] + current_proof})
        
    with open("data/results/rebuttal/rl_search_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nRL Search complete. Now verifying validities...")

if __name__ == "__main__":
    run_rl_search()
