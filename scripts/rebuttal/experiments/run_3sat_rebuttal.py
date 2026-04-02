import os
import json
import random
import torch
import numpy as np
import re
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from scipy.special import comb

# --- PURE PYTHON DPLL VERIFIER ---

def solve_sat_pure_python(clauses, variables):
    """Simple DPLL implementation for small 3-SAT formulas (up to 30 vars)."""
    def dpll(clauses, assignment):
        if not clauses: return True, assignment
        if any(not c for c in clauses): return False, None
        
        # Unit clause rule
        unit_clauses = [c for c in clauses if len(c) == 1]
        if unit_clauses:
            lit = unit_clauses[0][0]
            new_assignment = assignment.copy()
            new_assignment[abs(lit)] = (lit > 0)
            return dpll(simplify(clauses, lit), new_assignment)
            
        # Branching (heuristic: pick first variable)
        var = abs(clauses[0][0])
        # Try True
        res, final_assignment = dpll(simplify(clauses, var), {**assignment, var: True})
        if res: return True, final_assignment
        # Try False
        return dpll(simplify(clauses, -var), {**assignment, var: False})

    def simplify(clauses, literal):
        new_clauses = []
        for c in clauses:
            if literal in c: continue
            new_c = [l for l in c if l != -literal]
            new_clauses.append(new_c)
        return new_clauses

    is_sat, final_assignment = dpll(clauses, {})
    return is_sat, final_assignment

def verify_assignment(clauses, assignment):
    """Verifies that the assignment satisfies every clause."""
    if not assignment: return False
    for c in clauses:
        if not any((assignment.get(abs(lit)) == (lit > 0)) for lit in c): return False
    return True

def parse_assignment(text, n_vars):
    """Parses assignments like 'x1=T, x2=F' from model output."""
    assignment = {}
    try:
        cleaned = text.replace(" ", "").replace("\n", "").split("Assignment:")[-1]
        # Match pattern x[0-9]+=[TF]
        matches = re.findall(r"x(\d+)=([TF])", cleaned, re.IGNORECASE)
        for var_idx, val in matches:
            assignment[int(var_idx)] = (val.upper() == "T")
        return assignment if len(assignment) >= n_vars else None
    except: return None

# --- SPECTRAL UTILS ---

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
    except: return 0.0

def get_hfer_l21(model, tokenizer, text):
    """Extracts HFER at layer 21 for the provided text."""
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)
        h21 = compute_hfer_single_layer(outputs.attentions[21][0])
        del outputs
        torch.cuda.empty_cache()
    return h21

# --- EXPERIMENT ---

def calculate_pass_at_k(n, c, k):
    if n - c < k: return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)

def generate_3sat_instance(n_vars, ratio):
    m_clauses = int(n_vars * ratio)
    clauses = []
    while len(clauses) < m_clauses:
        variables = random.sample(range(1, n_vars + 1), 3)
        lits = [v if random.random() > 0.5 else -v for v in variables]
        if sorted(lits) not in clauses:
            clauses.append(sorted(lits))
    return clauses

def run_experiment():
    model_id = "meta-llama/Llama-3.1-8B-Instruct"
    print(f"Loading {model_id} (Config: output_attentions=True)...")
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map="auto", output_attentions=True)

    n_vars = 5 # Tractable for 8B
    ratios = [2.0, 4.26, 8.0] # Easy, Critical, Hard
    instances_per_ratio = 5
    
    # Few-shot examples
    few_shot = "Formula: (x1 OR x2 OR NOT x3) AND (NOT x1 OR x3 OR x4)\nAssignment: x1=T, x2=F, x3=T, x4=T\n\nFormula: (NOT x1 OR NOT x2 OR x3) AND (x2 OR x4 OR NOT x5)\nAssignment: x1=F, x2=T, x3=T, x4=F, x5=F\n\n"

    results = []
    output_path = "data/results/rebuttal/3sat_phase_results.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    for ratio in ratios:
        print(f"\n--- SWEEP: RATIO {ratio} ---")
        instances_collected = 0
        while instances_collected < instances_per_ratio:
            clauses = generate_3sat_instance(n_vars, ratio)
            is_sat, ground_truth = solve_sat_pure_python(clauses, n_vars)
            if not is_sat: continue  # Pass@k focus on SAT instances
            
            instances_collected += 1
            print(f"[{instances_collected}/{instances_per_ratio}] Ratio: {ratio}")
            
            clause_strs = []
            for c in clauses:
                terms = [f"{'' if lit > 0 else 'NOT '}x{abs(lit)}" for lit in c]
                clause_strs.append("(" + " OR ".join(terms) + ")")
            formula_str = " AND ".join(clause_strs)
            prompt = f"{few_shot}Formula: {formula_str}\nAssignment:"
            
            candidates = []
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            for i in range(8):
                with torch.no_grad():
                    out = model.generate(**inputs, max_new_tokens=64, do_sample=True, temperature=0.9, return_dict_in_generate=True, output_scores=True)
                    gen_text = tokenizer.decode(out.sequences[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
                    
                    assignment = parse_assignment("Assignment:" + gen_text, n_vars)
                    is_correct = verify_assignment(clauses, assignment) if assignment else False
                    hfer = get_hfer_l21(model, tokenizer, prompt + gen_text)
                    
                    t_scores = model.compute_transition_scores(out.sequences, out.scores, normalize_logits=True)
                    mean_lp = float(np.mean(t_scores[0].cpu().numpy()))
                    
                    candidates.append({"id": i, "is_correct": is_correct, "hfer": hfer, "mean_lp": mean_lp})
            
            correct_count = sum(1 for c in candidates if c['is_correct'])
            res_item = {
                "ratio": ratio,
                "correct_count": correct_count,
                "pass_at_1": calculate_pass_at_k(8, correct_count, 1),
                "pass_at_4": calculate_pass_at_k(8, correct_count, 4),
                "pass_at_8": calculate_pass_at_k(8, correct_count, 8),
                "bon_hfer_8": min(candidates, key=lambda x: x['hfer'])['is_correct'],
                "bon_lp_8": max(candidates, key=lambda x: x['mean_lp'])['is_correct']
            }
            results.append(res_item)
            with open(output_path, "w") as f:
                json.dump(results, f, indent=2)

    # Final summary
    print("\n" + "="*50)
    print("  3-SAT PHASE TRANSITION SUMMARY")
    print("="*50)
    for r in ratios:
        r_list = [res for res in results if res['ratio'] == r]
        p1 = np.mean([i['pass_at_1'] for i in r_list])
        p8 = np.mean([i['pass_at_8'] for i in r_list])
        bon_h = np.mean([1 if i['bon_hfer_8'] else 0 for i in r_list])
        print(f"Ratio {r}: Pass@1={p1:.2%}, Pass@8={p8:.2%}, BoN-HFER={bon_h:.2%}")
    print("="*50)

if __name__ == "__main__":
    run_experiment()
