import os
import json
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# --- REWARD UTILS ---

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
        # Using Layer 21 (middle-deep) as the reward sensor
        h21 = compute_hfer_single_layer(outputs.attentions[21][0])
        del outputs
        torch.cuda.empty_cache()
    return -h21 # Maximizing reward = minimizing HFER

# --- SEARCH AGENT ---

def run_rl_search_sim():
    model_id = "meta-llama/Llama-3.1-8B-Instruct"
    print(f"Loading {model_id} for RL Search Simulation...")
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map="auto", output_attentions=True)

    # 5 Hand-picked theorems that usually fail greedy search
    theorems = [
        {"name": "mathd_algebra_359", "statement": "theorem mathd_algebra_359 (y : \u211d) : (y + 6) * (y + 6) = y^2 + 12 * y + 36 := "},
        {"name": "mathd_numbertheory_66", "statement": "theorem mathd_numbertheory_66 : 194 % 11 = 7 := "},
        {"name": "mathd_algebra_140", "statement": "theorem mathd_algebra_140 (x : \u211d) (h0 : x = 2 * 24) : x = 48 := "},
        {"name": "mathd_algebra_15", "statement": "theorem mathd_algebra_15 (s : \u2115) (h0 : s = 2^1 + 2^2 + 2^3) : s = 14 := "},
        {"name": "mathd_numbertheory_3", "statement": "theorem mathd_numbertheory_3 : (12 * 7) % 5 = 4 := "}
    ]

    results = []
    print("\n--- SPECTRAL RL SEARCH (HFER-Guided) ---")
    
    for th in theorems:
        print(f"\nTheorem: {th['name']}")
        current_proof = "by "
        success = False
        
        # Step-by-step search (3 steps)
        for step in range(3):
            prompt = th['statement'] + current_proof
            # Generate 4 candidate next-steps (tactics)
            candidates = []
            with torch.no_grad():
                inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
                for _ in range(4):
                    out = model.generate(**inputs, max_new_tokens=16, do_sample=True, temperature=0.9, return_dict_in_generate=True, output_scores=True)
                    gen_text = tokenizer.decode(out.sequences[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).split('\n')[0]
                    
                    t_scores = model.compute_transition_scores(out.sequences, out.scores, normalize_logits=True)
                    mean_lp = float(np.mean(t_scores[0].cpu().numpy()))
                    
                    reward = get_hfer_reward(model, tokenizer, prompt + gen_text)
                    candidates.append({"text": gen_text, "reward": reward, "lp": mean_lp})
            
            # RL Selection: Max reward (minimizing HFER)
            best_candidate = max(candidates, key=lambda x: x['reward'])
            greedy_candidate = max(candidates, key=lambda x: x['lp'])
            
            print(f"Step {step}: Selected (HFER) '{best_candidate['text']}' vs (LP) '{greedy_candidate['text']}'")
            current_proof += best_candidate['text'] + " "
            
        results.append({"name": th['name'], "final_proof": current_proof})
        
    # Save results for final verification
    with open("data/results/rebuttal/hfer_rl_search.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSearch complete. Run verification next.")

if __name__ == "__main__":
    run_rl_search_sim()
