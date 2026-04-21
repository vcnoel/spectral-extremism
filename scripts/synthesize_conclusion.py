import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy.spatial.distance import mahalanobis
from sklearn.metrics import roc_auc_score
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# --- CONFIG ---
RESULTS_PATH = "results/spectra/rigorous_audit_results.json"
TIKHONOV_REG = 1e-6
MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"

def main():
    print("--- Loading N=500 Clean Sweep Results ---")
    with open(RESULTS_PATH, "r") as f:
        results = json.load(f)
    n = len(results)
    print(f"Loaded {n} matched pairs.")

    # 1. Extraction
    metric_keys = ["fiedler", "hfer", "smoothness", "entropy", "gini"]
    neu_l4 = np.array([[r["neutral_matched"]["l4_metrics"][m] for m in metric_keys] for r in results])
    rad_l4 = np.array([[r["radical_formal"]["l4_metrics"][m]  for m in metric_keys] for r in results])

    # 2. Mahalanobis D_M (L4 metrics 1-4)
    X_neu = neu_l4[:, :4]
    X_rad = rad_l4[:, :4]
    centroid = np.mean(X_neu, axis=0)
    cov = np.cov(X_neu, rowvar=False) + np.eye(X_neu.shape[1]) * TIKHONOV_REG
    cov_inv = np.linalg.pinv(cov)
    
    d_neu = [mahalanobis(x, centroid, cov_inv) for x in X_neu]
    d_rad = [mahalanobis(x, centroid, cov_inv) for x in X_rad]

    # 3. Tables (ASCII Only)
    print("\nTable 1: Layer 4 Descriptive Statistics (N=500, BOS Masked, Symmetrized)")
    print(f"{'Metric':<14} | {'Neutral (Matched)':<20} | {'Radical (Ghost)':<20}")
    print("-" * 65)
    labels = ["Fiedler", "HFER", "Smoothness", "Entropy", "Gini"]
    for i, lbl in enumerate(labels):
        m_neu, s_neu = np.mean(neu_l4[:,i]), np.std(neu_l4[:,i])
        m_rad, s_rad = np.mean(rad_l4[:,i]), np.std(rad_l4[:,i])
        ci_neu = 1.96 * s_neu / np.sqrt(n)
        ci_rad = 1.96 * s_rad / np.sqrt(n)
        print(f"{lbl:<14} | {m_neu:.4f} +/- {ci_neu:.4f} | {m_rad:.4f} +/- {ci_rad:.4f}")
    
    m_dn, s_dn = np.mean(d_neu), np.std(d_neu)
    m_dr, s_dr = np.mean(d_rad), np.std(d_rad)
    print(f"{'D_M':<14} | {m_dn:.4f} +/- {1.96*s_dn/np.sqrt(n):.4f} | {m_dr:.4f} +/- {1.96*s_dr/np.sqrt(n):.4f}")

    # 4. Classifier Performance
    y_true = [0]*n + [1]*n
    y_score = d_neu + d_rad
    auc = roc_auc_score(y_true, y_score)
    
    best_j = -1
    for t in np.linspace(min(y_score), max(y_score), 200):
        preds = (np.array(y_score) >= t).astype(int)
        tpr = np.sum(preds[n:] == 1) / n
        fpr = np.sum(preds[:n] == 1) / n
        if (tpr - fpr) > best_j: best_j = tpr - fpr
        
    print("\nTable 2: D_M Classifier Performance (Clean Data, N=500)")
    print(f"AUROC:       {auc:.4f}")
    print(f"J-Statistic: {best_j:.4f}")

    # 5. Outlier Extraction (Ghost Stability Check)
    print("\nGenerating Fractal Plot...")
    generate_fractal_plot(results)

    print("\nIdentifying Top 5 Outliers for Heatmaps...")
    results_sorted = sorted(results, key=lambda x: max([mahalanobis(np.array([x["radical_formal"]["l4_metrics"][m] for m in ["fiedler", "hfer", "smoothness", "entropy"]])[:4], centroid, cov_inv)]), reverse=True)
    top_5_ids = [r["radical_formal"]["text"][:50] for r in results_sorted[:5]]
    print("Top 5 Outlier Snippets:")
    for i, s in enumerate(top_5_ids): print(f"{i+1}: {s}...")
    
def generate_fractal_plot(results):
    os.makedirs("results/figures/rigorous", exist_ok=True)
    layers = range(len(results[0]["radical_formal"]["trajectory"]))
    plt.figure(figsize=(10,6))
    ax1 = plt.gca(); ax2 = ax1.twinx()
    
    for label, color, data_key in [("Neutral", "blue", "neutral_matched"), ("Radical", "red", "radical_formal")]:
        sm = np.mean([[tr["smoothness"] for tr in r[data_key]["trajectory"]] for r in results], axis=0)
        hf = np.mean([[tr["hfer"] for tr in r[data_key]["trajectory"]] for r in results], axis=0)
        ax1.plot(layers, sm, color=color, linestyle='-', label=f"Smoothness ({label})")
        ax2.plot(layers, hf, color=color, linestyle='--', label=f"HFER ({label})")
    
    plt.title("The Two-Stage Fractal of Radicalization (N=500 Clean Sweep)"); plt.savefig("results/figures/rigorous/fractal_clean.png")
    print("Visual saved to results/figures/rigorous/fractal_clean.png")

if __name__ == "__main__":
    main()
