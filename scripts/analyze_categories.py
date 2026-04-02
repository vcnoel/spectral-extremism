import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy import stats

def parse_args():
    parser = argparse.ArgumentParser(description="Spectral Category Analysis")
    parser.add_argument("--results", type=str, required=True, help="Path to extremism results JSON")
    parser.add_argument("--output-dir", type=str, default="results/figures/categories", help="Output directory")
    return parser.parse_args()

def compute_cohen_d(x1, x2):
    n1, n2 = len(x1), len(x2)
    s1, s2 = np.std(x1, ddof=1), np.std(x2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / (n1 + n2 - 2))
    return (np.mean(x1) - np.mean(x2)) / (pooled_std + 1e-9)

def run_categorical_analysis(args):
    os.makedirs(args.output_dir, exist_ok=True)
    
    with open(args.results, "r") as f:
        data = json.load(f)
    
    all_samples = data["radical"] + data["neutral"]
    
    # 1. Balanced Sampling (N=50 per class)
    # Mapping to the actual register-controlled categories in result JSON
    target_classes = [
        "jigsaw_radical", "jigsaw_neutral_informal", 
        "mhs_radical", "mhs_neutral", 
        "toxic_control", "wikipedia_formal"
    ]
    class_samples = {c: [] for c in target_classes}
    
    for s in all_samples:
        cat = s.get("category")
        if cat in class_samples:
            class_samples[cat].append(s)
            
    # Sample 50 each
    balanced_data = {}
    for cat in target_classes:
        samples = class_samples[cat]
        if not samples: continue
        
        if len(samples) >= 50:
            indices = np.random.choice(len(samples), 50, replace=False)
            balanced_data[cat] = [samples[i] for i in indices]
        else:
            print(f"Warning: Category {cat} has only {len(samples)} samples. Using all.")
            balanced_data[cat] = samples

    if not balanced_data:
        print("Error: No valid categories found for analysis.")
        return

    n_layers = len(list(balanced_data.values())[0][0]["trajectory"])
    n_classes = len(balanced_data)
    palette = sns.color_palette("colorblind", n_classes) 
    class_colors = {cat: palette[i] for i, cat in enumerate(balanced_data.keys())}
    
    # (a-b) Trajectory Plots with Std Bands
    metrics = ["hfer", "smoothness", "entropy"]
    for metric in metrics:
        plt.figure(figsize=(10, 6))
        for cat, samples in balanced_data.items():
            # Ensure all trajectories are same length
            trajs = []
            for s in samples:
                t_vals = [t[metric] for t in s["trajectory"] if t[metric] is not None]
                if len(t_vals) == n_layers:
                    trajs.append(t_vals)
            
            if not trajs: continue
            
            trajs = np.array(trajs)
            mean_traj = np.mean(trajs, axis=0)
            std_traj = np.std(trajs, axis=0)
            
            plt.plot(range(n_layers), mean_traj, label=cat, color=class_colors[cat], linewidth=3)
            plt.fill_between(range(n_layers), mean_traj - std_traj, mean_traj + std_traj, 
                             color=class_colors[cat], alpha=0.1)
            
        plt.title(f"Class Trajectories: {metric.capitalize()} (Mean ± Std)")
        plt.xlabel("Layer")
        plt.ylabel(metric)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, f"trajectory_{metric}.png"), dpi=300)
        print(f"Saved trajectory plot: {metric}")

    # Find "Best Layer" for Matrix and Scatter
    # Robust search: pick layer with max mean distance between any Radical and its same-source Neutral
    max_d = -1
    best_l = 0
    pairs = [
        ("mhs_radical", "mhs_neutral"),
        ("jigsaw_radical", "jigsaw_neutral_informal")
    ]
    
    valid_pairs = [(r, n) for r, n in pairs if r in balanced_data and n in balanced_data]
    if not valid_pairs:
        # Fallback to any radical vs any neutral
        rads = [c for c in balanced_data if "radical" in c]
        neus = [c for c in balanced_data if "neutral" in c or "formal" in c]
        if rads and neus: valid_pairs = [(rads[0], neus[0])]

    for l in range(n_layers):
        ds = []
        for r_cat, n_cat in valid_pairs:
            r_vals = [s["trajectory"][l]["smoothness"] for s in balanced_data[r_cat]]
            n_vals = [s["trajectory"][l]["smoothness"] for s in balanced_data[n_cat]]
            ds.append(abs(compute_cohen_d(r_vals, n_vals)))
        
        if ds:
            avg_d = np.mean(ds)
            if avg_d > max_d:
                max_d = avg_d
                best_l = l
    
    print(f"Best Layer for detailed analysis: L{best_l} (Avg Register-Controlled d={max_d:.2f})")

    # (c) Pairwise Cohen's d Matrix
    plt.figure(figsize=(10, 8))
    matrix_data = []
    class_names = list(balanced_data.keys())
    d_matrix = np.zeros((len(class_names), len(class_names)))
    
    for i, c1 in enumerate(class_names):
        for j, c2 in enumerate(class_names):
            v1 = [s["trajectory"][best_l]["smoothness"] for s in balanced_data[c1]]
            v2 = [s["trajectory"][best_l]["smoothness"] for s in balanced_data[c2]]
            d_matrix[i, j] = compute_cohen_d(v1, v2)
            
    df_d = pd.DataFrame(d_matrix, index=class_names, columns=class_names)
    sns.heatmap(df_d, annot=True, cmap="RdBu_r", center=0, fmt=".2f")
    plt.title(f"Pairwise Cohen's d at Best Layer (L{best_l}) - Smoothness")
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "pairwise_d_matrix.png"), dpi=300)
    print("Saved pairwise d matrix.")

    # (d) 2D Spectral Scatter Plot (HFER vs Smoothness)
    plt.figure(figsize=(10, 8))
    for cat, samples in balanced_data.items():
        x = [s["trajectory"][best_l]["hfer"] for s in samples]
        y = [s["trajectory"][best_l]["smoothness"] for s in samples]
        plt.scatter(x, y, label=cat, color=class_colors[cat], alpha=0.6, edgecolors='w', s=60)
        
    plt.title(f"2D Spectral Cluster at Best Layer (L{best_l})")
    plt.xlabel("HFER (X)")
    plt.ylabel("Smoothness (Y)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "spectral_2d_scatter.png"), dpi=300)
    print("Saved spectral 2D scatter plot.")
    
    # Print Matrix to stdout as requested
    print("\n" + "="*80)
    print(f"PAIRWISE COHEN'S d MATRIX (L{best_l}, Smoothness)")
    print("="*80)
    print(df_d.to_string())
    print("="*80 + "\n")

if __name__ == "__main__":
    args = parse_args()
    run_categorical_analysis(args)
