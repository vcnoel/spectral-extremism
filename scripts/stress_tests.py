import os
import torch
import numpy as np
import pandas as pd
import json
from tqdm import tqdm
from scipy.spatial.distance import mahalanobis
from scipy.linalg import eigh
from sklearn.metrics import roc_auc_score
from transformers import AutoModelForCausalLM, AutoTokenizer

def gini(array):
    array = array.flatten()
    if np.amin(array) < 0: array -= np.amin(array)
    array += 1e-12
    array = np.sort(array)
    index = np.arange(1, array.shape[0] + 1)
    n = array.shape[0]
    return ((np.sum((2 * index - n - 1) * array)) / (n * np.sum(array)))

def compute_metrics(A):
    D = np.diag(np.sum(A, axis=1))
    L = D - A
    D_inv_sqrt = np.diag(1.0 / (np.sqrt(np.diag(D)) + 1e-12))
    L_norm = np.eye(len(A)) - D_inv_sqrt @ A @ D_inv_sqrt
    evals = eigh(L_norm, eigvals_only=True)
    
    fiedler = evals[1] if len(evals) > 1 else 0.0
    hfer = np.sum(evals[evals > 1.0]) / (np.sum(evals) + 1e-12)
    smooth = np.sum(evals[evals < 0.5]) / (np.sum(evals) + 1e-12)
    ps = np.clip(evals, 0, None)
    ps = ps / (np.sum(ps) + 1e-12)
    entropy = -np.sum(ps * np.log(ps + 1e-12))
    epr = (np.sum(evals)**2) / (len(evals) * np.sum(evals**2) + 1e-12)
    return fiedler, hfer, smooth, entropy, epr

def detect_lang(text):
    text = text.lower()
    en_words = ['the', 'is', 'and', 'to', 'in']
    es_words = ['el', 'la', 'y', 'de', 'en', 'que']
    it_words = ['il', 'la', 'e', 'di', 'in', 'che']
    
    en_score = sum(1 for w in en_words if f" {w} " in f" {text} ")
    es_score = sum(1 for w in es_words if f" {w} " in f" {text} ")
    it_score = sum(1 for w in it_words if f" {w} " in f" {text} ")
    
    scores = {"EN": en_score, "ES": es_score, "IT": it_score}
    return max(scores, key=scores.get)

def run_stress_tests():
    print("--- FINAL FORENSIC PROOFS: STRESS TESTING THE MANIFOLD ---")
    model_name = "meta-llama/Llama-3.1-8B-Instruct"
    print(f"Loading {model_name} in 4-bit...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        device_map="auto", 
        load_in_4bit=True,
        torch_dtype=torch.float16,
        attn_implementation="eager"
    )

    df = pd.read_csv("results/advanced_stats.csv")
    extremists = df[df['label'] == 1].sort_values('dm', ascending=False).head(50)
    neutrals = df[df['label'] == 0].sort_values('dm', ascending=True).head(50)
    
    # Pre-compiled Formalized Texts from previous run
    with open("results/ghost_hunter.json", 'r') as f:
        ghost_data = json.load(f)
        # Note: ghost_hunter didn't save the actual texts! We'll just generate them quickly.

    # Fast Style-Transfer
    print("\n[1] Formalizing 50 Extremist Samples (Style-Transfer)...")
    formalized_texts = []
    original_texts = []
    for _, row in extremists.iterrows():
        orig = str(row['text'])
        prompt = f"Rewrite the following text into a formal, academic sentence evaluating the subject neutrally. Remove toxicity/slang:\nOriginal: {orig}\nRewritten:"
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=60, do_sample=False)
        out_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        formalized_texts.append(out_text)
        original_texts.append(orig)

    def extract_trajectory(texts):
        trajs = []
        for text in tqdm(texts, leave=False):
            inputs = tokenizer(text, return_tensors="pt").to(model.device)
            with torch.no_grad():
                out = model(**inputs, output_attentions=True)
            traj = []
            for l in range(len(out.attentions)):
                # mean over heads
                A = out.attentions[l][0].cpu().numpy().astype(np.float32).mean(axis=0)
                f, h, s, e, epr = compute_metrics(A)
                gini_val = gini(A)
                traj.append({"fiedler": f, "hfer": h, "smoothness": s, "entropy": e, "epr": epr, "gini": gini_val})
            trajs.append({"text": text, "traj": traj})
        return trajs

    print("\n[2] Executing Semantic Saturation Extraction (Across Model Layer Horizon)...")
    neutral_texts = [str(r['text']) for _, r in neutrals.iterrows()]
    neu_data = extract_trajectory(neutral_texts)
    for_data = extract_trajectory(formalized_texts)

    # Search for Inception Layer (Optimal Gini/Mahalanobis separation in Early Horizon L2-L8)
    L_INCEPTION = 6 # Llama-3.1-8B depth scaling -> typically L6/L7 correlates to L4 in 3B
    
    # Build L6 Centroid
    def get_vec(d, layer): return [d['traj'][layer]['fiedler'], d['traj'][layer]['hfer'], d['traj'][layer]['smoothness'], d['traj'][layer]['entropy']]
    
    X_neu_L6 = np.array([get_vec(d, L_INCEPTION) for d in neu_data], dtype=np.float64)
    X_neu_L6 += np.random.normal(0, 1e-7, size=X_neu_L6.shape)
    centroid = np.mean(X_neu_L6, axis=0)
    cov = np.cov(X_neu_L6, rowvar=False) + np.eye(4)*1e-5
    cov_inv = np.linalg.pinv(cov)
    
    def get_dm(x): return mahalanobis(x, centroid, cov_inv)
    
    dm_neu = [get_dm(x) for x in X_neu_L6]
    dm_for = [get_dm(get_vec(d, L_INCEPTION)) for d in for_data]
    
    auc = roc_auc_score([0]*50 + [1]*50, dm_neu + dm_for)
    print(f"-> Semantic Saturation Profile on Llama-3.1-8B (Layer {L_INCEPTION}) | ROC-AUC: {auc:.4f}")

    print("\n[3] Semantic Slicing Proof: The Ghost Ablation")
    # Sort formalized by Gini at Inception Layer
    for i, d in enumerate(for_data): d['id'] = i
    top_3 = sorted(for_data, key=lambda x: x['traj'][L_INCEPTION]['gini'], reverse=True)[:3]
    
    print("Ablating ideological anchors recursively via LLM...")
    ablated_texts = []
    for d in top_3:
        prompt = f"Identify the most polarizing or abstractly ideological word in this sentence and replace it with a harmless, boring, operational synonym. Preserve exact structure:\nText: {d['text']}\nRewritten:"
        inp = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inp, max_new_tokens=40, do_sample=False)
        abl = tokenizer.decode(out[0][inp.input_ids.shape[1]:], skip_special_tokens=True).strip()
        ablated_texts.append(abl)
        
    abl_data = extract_trajectory(ablated_texts)
    abl_dms = [get_dm(get_vec(d, L_INCEPTION)) for d in abl_data]
    for i in range(3):
        orig_dm = get_dm(get_vec(top_3[i], L_INCEPTION))
        print(f"Sample {i+1} | Pre-Ablation D_M: {orig_dm:.2f} -> Post-Ablation D_M: {abl_dms[i]:.2f}")

    print("\n[4] Translation Invariance Geodesic Mapping")
    langs = {"EN": [], "ES": [], "IT": []}
    for d in for_data:
        langs[detect_lang(d['text'])].append(get_vec(d, L_INCEPTION))
    
    c_en = np.mean(langs["EN"], axis=0) if langs["EN"] else centroid
    c_es = np.mean(langs["ES"], axis=0) if langs["ES"] else centroid
    c_it = np.mean(langs["IT"], axis=0) if langs["IT"] else centroid
    
    d_enes = np.linalg.norm(c_en - c_es)
    d_enit = np.linalg.norm(c_en - c_it)
    print(f"-> Geodesic Delta EN-ES: {d_enes:.4f} | EN-IT: {d_enit:.4f}")
    
    print("\n[5] Pseudo-Radical False Positive Shield")
    pseudo_texts = [
        "The academic theory of extremist radicalization emphasizes socioeconomic exclusion as the primary driver of online hostility.",
        "Social media moderation struggles to differentiate between protected political discourse and explicit ideological radicalism.",
        "This dataset investigates linguistic patterns associated with targeted online harassment campaigns.",
        "The sociology department published a paper analyzing the structural components of online radicalization algorithms.",
        "A critical review of censorship laws highlights the tension between free speech and containing political extremism."
    ]
    pseudo_data = extract_trajectory(pseudo_texts)
    pseudo_dms = [get_dm(get_vec(d, L_INCEPTION)) for d in pseudo_data]
    print(f"-> Pseudo-Radical False Positive Average D_M: {np.mean(pseudo_dms):.4f} (Threshold to flag > 10.0)")

    print("\nSTRESS TESTS COMPLETE.")

if __name__ == "__main__":
    run_stress_tests()
