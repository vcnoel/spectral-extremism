import os
import torch
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
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
    # A is [seq_len, seq_len] adjacency
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

def rewrite_text(model, tokenizer, text):
    prompt = f"Rewrite the following text into a very formal, neutral, and academic paragraph. Retain the core topic but remove all toxicity, slang, or emotional language.\n\nOriginal: {text}\n\nRewritten:"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=100, do_sample=False)
    out_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
    return out_text

def run():
    print("--- FORENSIC AUDIT OF INTENT INCEPTION ---")
    df = pd.read_csv("results/advanced_stats.csv")
    extremists = df[df['label'] == 1].sort_values('dm', ascending=False).head(50)
    neutrals = df[df['label'] == 0].sort_values('dm', ascending=True).head(50)
    
    model_name = "meta-llama/Llama-3.2-3B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.float16, attn_implementation="eager")
    
    results = {"neutral": [], "raw": [], "formalized": []}
    
    def process_sequence(text, category):
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model(**inputs, output_attentions=True)
            
        attentions = [a[0].cpu().numpy().astype(np.float32) for a in outputs.attentions]
        
        traj = []
        l4_gini = None
        l2_l6_mass = {} # attention masses for heatmap
        
        for l in range(len(attentions)):
            A = np.mean(attentions[l], axis=0) # [seq, seq]
            fv, hf, sm, en, epr = compute_metrics(A)
            
            if l == 4:
                l4_gini = gini(A)
                
            if 2 <= l <= 6:
                # Token sink mass (summing rows entering each column)
                sink_mass = np.mean(A, axis=0)
                l2_l6_mass[l] = sink_mass.tolist()
                
            traj.append({"layer": l, "fiedler": fv, "hfer": hf, "smoothness": sm, "entropy": en, "epr": epr})
            
        # extract tokens for heatmap plotting
        tokens = [tokenizer.decode([tid]) for tid in inputs.input_ids[0]]
        
        return {
            "text": text,
            "category": category,
            "trajectory": traj,
            "l4_gini": float(l4_gini) if l4_gini else 0,
            "l2_l6_mass": l2_l6_mass,
            "tokens": tokens
        }

    print("\nProcessing Neutral Baselines...")
    for _, row in tqdm(neutrals.iterrows(), total=len(neutrals)):
        results["neutral"].append(process_sequence(str(row['text']), "neutral"))
        
    print("\nProcessing Raw and Formalized Extremists...")
    for _, row in tqdm(extremists.iterrows(), total=len(extremists)):
        orig = str(row['text'])
        results["raw"].append(process_sequence(orig, "raw"))
        formal = rewrite_text(model, tokenizer, orig)
        results["formalized"].append(process_sequence(formal, "formalized"))
        
    # --- 1. Comparative Statistics Table (Layer 4) ---
    print("\n==============================================")
    print(" THE SMOKING GUN: L4 COMPARATIVE STATS MATRIX")
    print("==============================================")
    # Build L4 vectors to compute Mahalanobis Distance
    def get_l4_vec(item):
        t = item["trajectory"][4]
        return [t["fiedler"], t["hfer"], t["smoothness"], t["entropy"]]

    L4_neu = np.array([get_l4_vec(i) for i in results["neutral"]], dtype=np.float64)
    L4_raw = np.array([get_l4_vec(i) for i in results["raw"]], dtype=np.float64)
    L4_for = np.array([get_l4_vec(i) for i in results["formalized"]], dtype=np.float64)
    
    # Add robust jitter to prevent SVD collapse on constant parameters
    np.random.seed(42)
    L4_neu += np.random.normal(0, 1e-7, size=L4_neu.shape)
    
    centroid = np.mean(L4_neu, axis=0)
    cov = np.cov(L4_neu, rowvar=False) + np.eye(L4_neu.shape[1]) * 1e-6
    cov_inv = np.linalg.pinv(cov)
    
    def get_dms(X):
        return [mahalanobis(x, centroid, cov_inv) for x in X]
        
    neu_dms = get_dms(L4_neu)
    raw_dms = get_dms(L4_raw)
    for_dms = get_dms(L4_for)
    
    # Store DMs
    for idx in range(50):
        results["neutral"][idx]["l4_dm"] = neu_dms[idx]
        results["raw"][idx]["l4_dm"] = raw_dms[idx]
        results["formalized"][idx]["l4_dm"] = for_dms[idx]

    df_stats = []
    for cat in ["neutral", "raw", "formalized"]:
        g = np.mean([i["l4_gini"] for i in results[cat]])
        f = np.mean([i["trajectory"][4]["fiedler"] for i in results[cat]])
        e = np.mean([i["trajectory"][4]["epr"] for i in results[cat]])
        d = np.mean([i["l4_dm"] for i in results[cat]])
        df_stats.append({"Category": cat.upper(), "D_M": d, "Gini": g, "Fiedler": f, "EPR": e})
        
    stats_df = pd.DataFrame(df_stats)
    print(stats_df.to_string(index=False))
    
    # Measure ROC-AUC on Formalized vs Neutral at L4
    labels = [0]*50 + [1]*50
    scores = neu_dms + for_dms
    auc = roc_auc_score(labels, scores)
    
    # Youden's J-statistic
    best_j, best_thresh = -1, 0
    thresholds = np.linspace(min(scores), max(scores), 100)
    for t in thresholds:
        preds = (np.array(scores) >= t).astype(int)
        tpr = np.sum((preds[50:] == 1)) / 50.0 # class 1
        fpr = np.sum((preds[:50] == 1)) / 50.0 # class 0
        if (tpr - fpr) > best_j:
            best_j, best_thresh = tpr - fpr, t
            
    print(f"\nForensic Formalized vs Neutral L4 ROC-AUC: {auc:.4f}")
    print(f"Optimal 'Topological Perimeter' L4 D_M Threshold: {best_thresh:.4f} (J-Stat: {best_j:.4f})")

    os.makedirs("results/figures/forensic", exist_ok=True)
    
    # --- 2. Dual-Axis Fracture Plot ---
    layers = list(range(28))
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()
    
    colors = {"neutral": "blue", "raw": "red", "formalized": "gold"}
    labels_map = {"neutral": "Pure Neutral (Formal)", "raw": "Raw Radical (Slang)", "formalized": "Formalized Radical (Ghost)"}
    
    for cat in ["neutral", "raw", "formalized"]:
        smooth_means = [np.mean([i["trajectory"][L]["smoothness"] for i in results[cat]]) for L in layers]
        hfer_means = [np.mean([i["trajectory"][L]["hfer"] for i in results[cat]]) for L in layers]
        
        ax1.plot(layers, smooth_means, color=colors[cat], linestyle='-', linewidth=2, label=f"Smoothness [{labels_map[cat]}]")
        ax2.plot(layers, hfer_means, color=colors[cat], linestyle='--', linewidth=2, label=f"HFER [{labels_map[cat]}]")
        
    ax1.axvspan(2, 5, color='orange', alpha=0.15, label='Early Horizon Fracture (L2-L5)')
    ax1.axvspan(22, 27, color='purple', alpha=0.15, label='Sociolinguistic Sink (L22-L27)')
    ax1.set_xlabel('Transformer Layer Depth')
    ax1.set_ylabel('Smoothness (\u03B7)', color='black')
    ax2.set_ylabel('HFER (High-Frequency Energy)', color='black')
    plt.title("The Two-Stage Fractal of Radicalization Geometry", fontweight='bold')
    
    # combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)
    plt.tight_layout()
    plt.savefig("results/figures/forensic/dual_fracture_plot.png", dpi=300, bbox_inches='tight')
    print("Saved dual_fracture_plot.png")
    
    # --- 3. The Ghost Heatmap ---
    formalized_sorted = sorted(results["formalized"], key=lambda k: k["l4_gini"], reverse=True)
    top_3 = formalized_sorted[:3]
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 12))
    for i, item in enumerate(top_3):
        ax = axes[i]
        mass_matrix = [item["l2_l6_mass"][L] for L in [2, 3, 4, 5, 6]]
        sns.heatmap(mass_matrix, cmap='YlOrRd', ax=ax, cbar=(i==0), yticklabels=["L2", "L3", "L4", "L5", "L6"])
        ax.set_xticks(np.arange(len(item["tokens"])) + 0.5)
        ax.set_xticklabels(item["tokens"], rotation=45, ha='right', fontsize=8)
        ax.set_title(f"Ghost Heatmap {i+1} : L4 Gini={item['l4_gini']:.3f}")
        
    plt.tight_layout()
    plt.savefig("results/figures/forensic/ghost_attention.png", dpi=300)
    print("Saved ghost_attention.png heatmaps.")
    print("Forensic Audit complete.")

if __name__ == "__main__":
    run()
