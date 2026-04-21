import os
import torch
import numpy as np
import pandas as pd
import json
from tqdm import tqdm
from scipy.spatial.distance import mahalanobis
from scipy.linalg import eigh
from sklearn.metrics import roc_auc_score
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# --- Spectral Diagnostic Functions ---

def gini(array):
    array = array.flatten()
    if np.amin(array) < 0: array -= np.amin(array)
    array += 1e-12
    array = np.sort(array)
    index = np.arange(1, array.shape[0] + 1)
    n = array.shape[0]
    return ((np.sum((2 * index - n - 1) * array)) / (n * np.sum(array)))

def compute_metrics(A):
    # Ensure float64 for numerical stability in spectral decomp
    A = A.astype(np.float64)
    D = np.diag(np.sum(A, axis=1))
    D_inv_sqrt = np.diag(1.0 / (np.sqrt(np.diag(D)) + 1e-12))
    L_norm = np.eye(len(A)) - D_inv_sqrt @ A @ D_inv_sqrt
    
    # Add small diagonal shift to L_norm for stability
    L_norm += np.eye(len(A)) * 1e-8
    
    try:
        evals = eigh(L_norm, eigvals_only=True)
    except:
        # Fallback if eigh fails
        return 0.0, 0.0, 0.0, 0.0, 0.0
        
    fiedler = evals[1] if len(evals) > 1 else 0.0
    hfer = np.sum(evals[evals > 1.0]) / (np.sum(evals) + 1e-12)
    smooth = np.sum(evals[evals < 0.5]) / (np.sum(evals) + 1e-12)
    
    ps = np.clip(evals, 1e-12, None)
    ps = ps / np.sum(ps)
    entropy = -np.sum(ps * np.log(ps))
    
    epr = (np.sum(evals)**2) / (len(evals) * np.sum(evals**2) + 1e-12)
    return fiedler, hfer, smooth, entropy, epr

# --- Model Logic ---

def rewrite_text(model, tokenizer, text):
    prompt = f"Rewrite the following text into a very formal, neutral, and academic paragraph. Retain the core topic but remove all toxicity, slang, or emotional language.\n\nOriginal: {text}\n\nRewritten:"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=100, do_sample=False)
    out_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
    return out_text

def extract_l4_metrics(model, tokenizer, texts, desc="Extracting"):
    all_metrics = []
    for text in tqdm(texts, desc=desc):
        inputs = tokenizer(str(text), return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model(**inputs, output_attentions=True)
        
        # Pull Layer 4 attention only to save memory
        A_raw = outputs.attentions[4][0].cpu().numpy().astype(np.float32)
        A = np.mean(A_raw, axis=0) # Head average
        
        # Spectral Diagnostics
        fv, hf, sm, en, epr = compute_metrics(A)
        g = gini(A)
        
        all_metrics.append({
            "fiedler": fv, "hfer": hf, "smoothness": sm, "entropy": en, "epr": epr, "gini": g
        })
    return all_metrics

# --- Main Execution ---

def run():
    print("--- COMMENCING RIGOR & GENERALIZATION SWEEP ---")
    
    # 1. Load Curated Data
    with open('data/curated_rigor_sweep.json', 'r') as f:
        data = json.load(f)
        
    # 2. Model Setup (4-bit)
    model_name = "meta-llama/Llama-3.2-3B-Instruct"
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config, device_map="auto", attn_implementation="eager")
    
    # ---------------------------------------------------------
    # TASK 1: OOD CROSS-LANGUAGE TEST (English -> ES/IT)
    # ---------------------------------------------------------
    print("\n[TASK 1] English Centroid Isolation...")
    en_neu_metrics = extract_l4_metrics(model, tokenizer, [d['text'] for d in data['en_neutral_centroid']], "EN Neutral")
    
    L4_en = np.array([[m['fiedler'], m['hfer'], m['smoothness'], m['entropy']] for m in en_neu_metrics], dtype=np.float64)
    centroid = np.mean(L4_en, axis=0)
    # Tikhonov Regularization (1e-6)
    cov = np.cov(L4_en, rowvar=False) + np.eye(4) * 1e-6
    cov_inv = np.linalg.inv(cov)
    
    def get_dm(metrics_list):
        vecs = np.array([[m['fiedler'], m['hfer'], m['smoothness'], m['entropy']] for m in metrics_list])
        return [mahalanobis(v, centroid, cov_inv) for v in vecs]

    print("[TASK 1] Testing Language Invariance (ES/IT)...")
    # First formalize them to remove keyword bias
    es_texts = [rewrite_text(model, tokenizer, d['text']) for d in data['es_radical']]
    it_texts = [rewrite_text(model, tokenizer, d['text']) for d in data['it_radical']]
    
    es_metrics = extract_l4_metrics(model, tokenizer, es_texts, "ES Formal")
    it_metrics = extract_l4_metrics(model, tokenizer, it_texts, "IT Formal")
    
    es_dms = get_dm(es_metrics)
    it_dms = get_dm(it_metrics)
    neu_dms = get_dm(en_neu_metrics) # self-check
    
    # ROC-AUC: ES Radical vs EN Neutral
    labels_es = [0]*len(neu_dms) + [1]*len(es_dms)
    scores_es = neu_dms + es_dms
    auc_es = roc_auc_score(labels_es, scores_es)
    
    labels_it = [0]*len(neu_dms) + [1]*len(it_dms)
    scores_it = neu_dms + it_dms
    auc_it = roc_auc_score(labels_it, scores_it)
    
    print(f"\nTask 1 RESULTS:")
    print(f"| Train Lang | Test Lang | AUROC |")
    print(f"|------------|-----------|-------|")
    print(f"| English    | Spanish   | {auc_es:.4f} |")
    print(f"| English    | Italian   | {auc_it:.4f} |")

    # ---------------------------------------------------------
    # TASK 2: REAL HUMAN MANIFESTO TEST (Length-Matched)
    # ---------------------------------------------------------
    print("\n[TASK 2] Human Manifesto Validation (Length-Matched)...")
    human_rad_metrics = extract_l4_metrics(model, tokenizer, [d['text'] for d in data['human_radical']], "Human Radical")
    human_neu_metrics = extract_l4_metrics(model, tokenizer, [d['text'] for d in data['human_neutral']], "Human Neutral")
    
    rad_dms = get_dm(human_rad_metrics)
    neu_human_dms = get_dm(human_neu_metrics)
    
    print(f"\nTask 2 RESULTS (Avg Metrics):")
    print(f"| Group         | Avg D_M | Avg Gini | Avg Fiedler |")
    print(f"|---------------|---------|----------|-------------|")
    print(f"| Human Radical | {np.mean(rad_dms):.2f}  | {np.mean([m['gini'] for m in human_rad_metrics]):.4f}   | {np.mean([m['fiedler'] for m in human_rad_metrics]):.4f}      |")
    print(f"| Human Neutral | {np.mean(neu_human_dms):.2f}    | {np.mean([m['gini'] for m in human_neu_metrics]):.4f}   | {np.mean([m['fiedler'] for m in human_neu_metrics]):.4f}      |")

    # ---------------------------------------------------------
    # TASK 3: SCALE TO N=1,000 (Law of Large Numbers)
    # ---------------------------------------------------------
    print("\n[TASK 3] Scaling to N=1,000...")
    # Select 1,000 radical samples pool
    pool = [d['text'] for d in data['n_1000_pool']]
    
    print("Formalizing pool (Batch 1,000)...")
    formalized_pool = []
    for t in tqdm(pool, desc="Batch Style Transfer"):
        formalized_pool.append(rewrite_text(model, tokenizer, t))
        
    print("Extracting L4 Metrics...")
    pool_metrics = extract_l4_metrics(model, tokenizer, formalized_pool, "N=1000 Extract")
    pool_dms = get_dm(pool_metrics)
    
    # 95% Confidence Intervals
    def get_ci(vals):
        mean = np.mean(vals)
        std = np.std(vals)
        h = 1.96 * (std / np.sqrt(len(vals)))
        return mean, h

    dm_m, dm_h = get_ci(pool_dms)
    gi_m, gi_h = get_ci([m['gini'] for m in pool_metrics])
    fi_m, fi_h = get_ci([m['fiedler'] for m in pool_metrics])
    
    print(f"\nTask 3 RESULTS (N=1,000 Grand Matrix):")
    print(f"| Metric    | Mean     | 95% CI (+/-) |")
    print(f"|-----------|----------|-------------|")
    print(f"| L4 D_M    | {dm_m:.4f} | {dm_h:.4f}      |")
    print(f"| L4 Gini   | {gi_m:.4f} | {gi_h:.4f}      |")
    print(f"| L4 Fiedler| {fi_m:.4f} | {fi_h:.4f}      |")

    print("\n--- RIGOR SWEEP COMPLETE ---")

if __name__ == "__main__":
    run()
