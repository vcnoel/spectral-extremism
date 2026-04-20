import os
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from scipy.spatial.distance import mahalanobis
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, roc_auc_score
from transformers import AutoTokenizer

try:
    import pingouin as pg
except ImportError:
    pass

try:
    import umap.umap_ as umap
except ImportError:
    pass

def load_data(json_path):
    print(f"Loading {json_path}...")
    with open(json_path, 'r') as f: return json.load(f)

def build_140d_vectors(data):
    X, labels, categories, texts, ids = [], [], [], [], []
    for label_key, label_val in [("radical", 1), ("neutral", 0)]:
        for s in data[label_key]:
            traj = s["trajectory"]
            if len(traj) != 28: continue
            
            vec = []
            for t in traj:
                fv = t["fiedler_value"] if t["fiedler_value"] is not None else 0.0
                hf = t["hfer"] if t["hfer"] is not None else 0.0
                sm = t["smoothness"] if t["smoothness"] is not None else 0.0
                en = t["entropy"] if t["entropy"] is not None else 1e-5
                sc = fv / (en + 1e-9) # Spectral Cohesion
                vec.extend([fv, hf, sm, en, sc])
            
            if not np.any(np.isnan(vec)):
                X.append(vec)
                labels.append(label_val)
                categories.append(s.get("category", "unknown"))
                texts.append(s.get("text", ""))
                ids.append(s.get("id", str(len(ids))))
                
    return np.array(X), np.array(labels), np.array(categories), np.array(texts), np.array(ids)

def run_stats_mode(args):
    data = load_data(args.dataset)
    X, labels, categories, texts, ids = build_140d_vectors(data)
    print(f"Constructed vectors: {X.shape}")
    
    # 1. Platonic Gap (Mahalanobis Distance)
    print("\n--- 1. High-Dimensional 'Platonic Gap' Analysis ---")
    benign_idx = (labels == 0)
    X_benign = X[benign_idx]
    centroid = np.mean(X_benign, axis=0)
    
    # Shrinkage covariance or Moore-Penrose pseudo-inverse since N=1000, D=140
    cov = np.cov(X_benign, rowvar=False)
    cov_inv = np.linalg.pinv(cov, rcond=1e-5)
    
    dm_scores = []
    for i in range(len(X)):
        dist = mahalanobis(X[i], centroid, cov_inv)
        dm_scores.append(dist)
    dm_scores = np.array(dm_scores)
    
    auc = roc_auc_score(labels, dm_scores)
    print(f"ROC-AUC based on D_M: {auc:.4f}")
    
    # Youden's J-statistic
    best_j, best_thresh = -1, 0
    thresholds = np.linspace(dm_scores.min(), dm_scores.max(), 100)
    for t in thresholds:
        preds = (dm_scores >= t).astype(int)
        tpr = np.sum((preds == 1) & (labels == 1)) / np.sum(labels == 1)
        fpr = np.sum((preds == 1) & (labels == 0)) / np.sum(labels == 0)
        j_stat = tpr - fpr
        if j_stat > best_j:
            best_j, best_thresh = j_stat, t
    print(f"Optimal Youden's J-statistic: {best_j:.4f} at Threshold {best_thresh:.4f}")
    
    # 2. Spectral Taxonomy (Clustering)
    print("\n--- 2. Unsupervised Machine Taxonomies ---")
    best_k, best_sil = -1, -1
    best_clusters = None
    for k in range(2, 7):
        try:
            # We use KMeans since 140D SpectralClustering can be slow/unstable locally
            from sklearn.cluster import SpectralClustering
            km = KMeans(n_clusters=k, random_state=42)
            c = km.fit_predict(X)
            sil = silhouette_score(X, c)
            if sil > best_sil:
                best_sil, best_k, best_clusters = sil, k, c
        except Exception as e:
            pass
            
    print(f"Optimal Clusters (k): {best_k} (Silhouette: {best_sil:.3f})")
    df = pd.DataFrame({"label": labels, "category": categories, "cluster": best_clusters, "dm": dm_scores, "text": texts, "id": ids})
    print("Cluster compositions (Cluster x Category):")
    ct = pd.crosstab(df['cluster'], df['category'])
    print(ct)
    
    # Calculate Geodesic Delta across structural languages (centroids of es vs en vs it in 140D)
    print("\n--- Geodesic Delta (Languages) ---")
    lang_centroids = {}
    for lang in ["en", "es", "it"]:
        mask = df['category'].str.contains(f"_{lang}_")
        if mask.sum() > 0:
            lang_centroids[lang] = np.mean(X[mask], axis=0)
    if len(lang_centroids) >= 2:
        for l1 in lang_centroids:
            for l2 in lang_centroids:
                if l1 < l2:
                    dist = np.linalg.norm(lang_centroids[l1] - lang_centroids[l2])
                    print(f" Delta {l1} vs {l2}: {dist:.2f}")

    # 3. Confound Control
    print("\n--- 3. Confound Control (Length Partial Correlation) ---")
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
    tokens_len = [len(tokenizer.encode(t)) for t in tqdm(texts, desc="Tokenizing")]
    df['seq_len'] = tokens_len
    
    # Layer 25 HFER is idx 25*5 + 1 = 126
    hfer_25 = X[:, 25*5 + 1]
    df['hfer_25'] = hfer_25
    
    try:
        pcorr = pg.partial_corr(data=df, x='hfer_25', y='label', covar='seq_len')
        print("Partial Correlation (HFER L25 ~ Label | Length):")
        print(pcorr[['r', 'p-val']])
    except Exception as e:
        print(f"Pingouin partial corr failed: {e}")
        
    df.to_csv("results/advanced_stats.csv", index=False)
    print("Saved stats dataframe to results/advanced_stats.csv")
    
    # Visuals
    print("\n--- Generating Visuals ---")
    os.makedirs("results/figures/advanced", exist_ok=True)
    
    # Phase Portrait (L25 Fiedler vs HFER)
    fiedler_25 = X[:, 25*5 + 0]
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=fiedler_25, y=hfer_25, hue=categories, style=labels, palette="tab20", s=60, alpha=0.8)
    plt.axhline(np.mean(hfer_25[labels==0]), color='blue', linestyle='--', alpha=0.5)
    plt.axvline(np.mean(fiedler_25[labels==0]), color='blue', linestyle='--', alpha=0.5)
    plt.title("Spectral Phase Portrait (Layer 25 Singularity)", fontweight='bold')
    plt.xlabel("Algebraic Connectivity (Fiedler Value $\lambda_2$)")
    plt.ylabel("High-Frequency Energy Ratio (HFER)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig("results/figures/advanced/phase_portrait.png", dpi=300)
    print("Saved Phase Portrait.")
    
    # UMAP 3D
    try:
        reducer = umap.UMAP(n_components=3, random_state=42)
        X_umap = reducer.fit_transform(X)
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        rad_mask = (labels == 1)
        ax.scatter(X_umap[~rad_mask, 0], X_umap[~rad_mask, 1], X_umap[~rad_mask, 2], 
                   c='blue', label='Neutral/Benign', alpha=0.3, s=20)
        ax.scatter(X_umap[rad_mask, 0], X_umap[rad_mask, 1], X_umap[rad_mask, 2], 
                   c='red', label='Extremist', alpha=0.7, s=40, marker='^')
                   
        ax.set_title("3D Topological Manifold Pathologies", fontweight='bold')
        ax.legend()
        plt.tight_layout()
        plt.savefig("results/figures/advanced/manifold_umap_3d.png", dpi=300)
        print("Saved 3D UMAP Manifold.")
    except Exception as e:
        print(f"UMAP visualization failed: {e}")

import torch
from transformers import AutoModelForCausalLM
from spectral_trust import GSPDiagnosticsFramework, GSPConfig

def calculate_epr(attentions):
    import networkx as nx
    from scipy.linalg import eigh
    # attentions shape: [layers, heads, seq_len, seq_len]
    epr_per_layer = []
    # Just take mean over heads to form one graph per layer
    for layer_idx in range(len(attentions)):
        A = np.mean(attentions[layer_idx], axis=0) # [seq_len, seq_len]
        D = np.diag(np.sum(A, axis=1))
        L = D - A
        # Normalized Laplacian
        D_inv_sqrt = np.diag(1.0 / (np.sqrt(np.diag(D)) + 1e-12))
        L_norm = np.eye(len(A)) - D_inv_sqrt @ A @ D_inv_sqrt
        evals = eigh(L_norm, eigvals_only=True)
        # EPR formula
        epr = (np.sum(evals)**2) / (len(evals) * np.sum(evals**2) + 1e-12)
        epr_per_layer.append(epr)
    return epr_per_layer

def rewrite_text(model, tokenizer, text):
    prompt = f"Rewrite the following text into a very formal, neutral, and academic paragraph. Retain the core topic but remove all toxicity, slang, or emotional language.\n\nOriginal: {text}\n\nRewritten:"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=100, do_sample=False)
    out_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
    return out_text

def run_nuclear_mode(args):
    print("\n--- 4. Focused Nuclear EPR Analysis & Style Transfer ---")
    df = pd.read_csv("results/advanced_stats.csv")
    
    extremists = df[df['label'] == 1].sort_values('dm', ascending=False).head(50)
    neutrals = df[df['label'] == 0].sort_values('dm', ascending=True).head(50)
    print(f"Loaded Top 50 Extremists (Mean D_M: {extremists['dm'].mean():.2f})")
    print(f"Loaded Top 50 Benign Neutrals (Mean D_M: {neutrals['dm'].mean():.2f})")
    
    print("\nLoading Llama-3.2-3B-Instruct for Style-Transfer and Spectral Extraction...")
    model_name = "meta-llama/Llama-3.2-3B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.float16, attn_implementation="eager")
    
    extracted_data = []

    # Calculate EPR for Neutrals
    print("Extracting EPR across 50 Neutral samples...")
    for idx, row in tqdm(neutrals.iterrows(), total=len(neutrals)):
        text = str(row['text'])
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model(**inputs, output_attentions=True)
            atts = [a[0].cpu().numpy() for a in outputs.attentions]
            epr_traj = calculate_epr(atts)
            extracted_data.append({"type": "neutral", "epr": epr_traj})
            
    # Style Transfer & Compare EPR for Extremists
    print("Processing Top 50 Extremists (Original vs Adversarial Style-Transfer)...")
    for idx, row in tqdm(extremists.iterrows(), total=len(extremists)):
        orig_text = str(row['text'])
        # Original
        inputs = tokenizer(orig_text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model(**inputs, output_attentions=True)
            atts = [a[0].cpu().numpy() for a in outputs.attentions]
            orig_epr = calculate_epr(atts)
            extracted_data.append({"type": "extremist", "epr": orig_epr})
        
        # Style Transfer
        adv_text = rewrite_text(model, tokenizer, orig_text)
        # Extract Adversarial
        inputs_adv = tokenizer(adv_text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs_adv = model(**inputs_adv, output_attentions=True)
            atts_adv = [a[0].cpu().numpy() for a in outputs_adv.attentions]
            adv_epr = calculate_epr(atts_adv)
            extracted_data.append({"type": "transfer", "epr": adv_epr})
            
    # Visualize & Save
    with open("results/advanced_nuclear.json", "w") as f:
        json.dump(extracted_data, f, indent=2)
        
    print("\nNuclear EPR Extraction Complete.")
    
    # Simple summary of Layer 25 EPR
    neu_epr = np.mean([x["epr"][25] for x in extracted_data if x["type"] == "neutral"])
    ext_epr = np.mean([x["epr"][25] for x in extracted_data if x["type"] == "extremist"])
    tra_epr = np.mean([x["epr"][25] for x in extracted_data if x["type"] == "transfer"])
    print("--- Layer 25 EPR Summary ---")
    print(f"Neutral Baseline: {neu_epr:.4f}")
    print(f"Original Extremist: {ext_epr:.4f}")
    print(f"Style-Transferred Extremist: {tra_epr:.4f}")
    
    if abs(tra_epr - ext_epr) < abs(tra_epr - neu_epr):
        print("=> VALIDATION SUCCESS: The Spectral Collapse (Low EPR) persists in neutral style. The signature is structurally invariant to register bloat.")
    else:
        print("=> VALIDATION FAILED: The signature disappeared when style was removed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["stats", "nuclear"], required=True)
    parser.add_argument("--dataset", type=str, default="results/spectra/extremism_results_Llama-3.2-3B-Instruct.json")
    args = parser.parse_args()
    
    if args.mode == "stats":
        run_stats_mode(args)
    elif args.mode == "nuclear":
        run_nuclear_mode(args)
