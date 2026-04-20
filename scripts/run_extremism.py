import os
import json
import glob
import traceback
import argparse
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy import stats
from tqdm import tqdm
from spectral_trust import GSPDiagnosticsFramework, GSPConfig
from statsmodels.stats.multitest import multipletests
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def parse_args():
    parser = argparse.ArgumentParser(description="Spectral Extremism Experiment CLI")
    parser.add_argument("--model", type=str, required=True, help="HuggingFace model name")
    parser.add_argument("--dataset", type=str, default="data/extremism_dataset.json", help="Path to dataset JSON")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device")
    parser.add_argument("--offline", action="store_true", help="Local only")
    parser.add_argument("--plot-only", action="store_true", help="Plot only")
    parser.add_argument("--results-file", type=str, default=None, help="Results path")
    parser.add_argument("--load-in-8bit", action="store_true")
    parser.add_argument("--load-in-4bit", action="store_true")
    return parser.parse_args()

def compute_t_stats(radical, neutral):
    """Exhaustive stats for the final report."""
    if len(radical) < 2 or len(neutral) < 2: return None
    mu_r, std_r = np.mean(radical), np.std(radical)
    mu_n, std_n = np.mean(neutral), np.std(neutral)
    try: _, p_mw = stats.mannwhitneyu(radical, neutral, alternative='two-sided')
    except: p_mw = 1.0
    pooled_std = np.sqrt(((len(radical)-1)*std_r**2 + (len(neutral)-1)*std_n**2) / (len(radical)+len(neutral)-2))
    d = (mu_r - mu_n) / pooled_std if pooled_std > 0 else 0
    y_true = [1]*len(radical) + [0]*len(neutral)
    scores = list(radical) + list(neutral)
    try:
        auroc = roc_auc_score(y_true, scores)
        if auroc < 0.5: auroc = 1 - auroc # Magnitude
        auprc = average_precision_score(y_true, scores) # Simple approx
    except: auroc, auprc = 0.5, 0.0
    # Best Acc/F1
    best_f1, best_acc, best_thresh, best_dir = -1, 0, 0, ">="
    thresholds = np.linspace(min(scores), max(scores), 100)
    for t in thresholds:
        for direction in [">=", "<="]:
            y_pred = [1 if (s >= t if direction == ">=" else s <= t) else 0 for s in scores]
            p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
            acc = np.mean(np.array(y_pred) == np.array(y_true))
            if f1 > best_f1:
                best_f1, best_acc, best_thresh, best_dir = f1, acc, t, direction
    return {"d": d, "auroc": auroc, "auprc": auprc, "acc": best_acc, "f1": best_f1, "p_mw": p_mw, "thresh": best_thresh, "direction": best_dir}

def compute_t_stats_simple(vals, y):
    """Fast stats for inner CV loop."""
    valid_data = [(v, target) for v, target in zip(vals, y) if v is not None]
    if not valid_data: return {"acc": 0.5, "thresh": 0, "direction": ">="}
    
    scores = np.array([v for v, _ in valid_data])
    y_true = np.array([t for _, t in valid_data])
    
    best_acc, best_thresh, best_dir = -1, 0, ">="
    thresholds = np.linspace(min(scores), max(scores), 50)
    for t in thresholds:
        for direction in [">=", "<="]:
            y_pred = (scores >= t) if direction == ">=" else (scores <= t)
            acc = np.mean(y_pred == y_true)
            if acc > best_acc: best_acc, best_thresh, best_dir = acc, t, direction
    return {"acc": best_acc, "thresh": best_thresh, "direction": best_dir}

def compute_nested_cv(data):
    mhs_samples = [s for key in ["radical", "neutral"] for s in data[key] if s.get("category") in ["mhs_radical", "mhs_neutral"]]
    if len(mhs_samples) < 40: return 0.0
    X, y = mhs_samples, np.array([1 if s.get("category") == "mhs_radical" else 0 for s in mhs_samples])
    metrics = ["fiedler_value", "hfer", "smoothness", "entropy"]
    n_layers = len(X[0]["trajectory"])
    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = []
    for train_idx, test_idx in outer_cv.split(X, y):
        X_tr_o, y_tr_o = [X[i] for i in train_idx], y[train_idx]
        X_te_o, y_te_o = [X[i] for i in test_idx], y[test_idx]
        inner_cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
        best_inner_acc, best_params = -1, None
        for m in metrics:
            for l in range(n_layers):
                inner_accs = []
                for tr_i, val_i in inner_cv.split(X_tr_o, y_tr_o):
                    X_it, y_it = [X_tr_o[i] for i in tr_i], y_tr_o[tr_i]
                    X_iv, y_iv = [X_tr_o[i] for i in val_i], y_tr_o[val_i]
                    
                    res = compute_t_stats_simple([s["trajectory"][l][m] for s in X_it], y_it)
                    
                    # Filter val set for non-None
                    v_data = [(s["trajectory"][l][m], target) for s, target in zip(X_iv, y_iv) if s["trajectory"][l][m] is not None]
                    if not v_data: continue
                    
                    y_pred = [1 if (v >= res["thresh"] if res["direction"] == ">=" else v <= res["thresh"]) else 0 for v, _ in v_data]
                    y_true_v = [t for _, t in v_data]
                    inner_accs.append(np.mean(np.array(y_pred) == np.array(y_true_v)))
                
                if inner_accs and np.mean(inner_accs) > best_inner_acc:
                    best_inner_acc = np.mean(inner_accs)
                    res_train = compute_t_stats_simple([s["trajectory"][l][m] for s in X_tr_o], y_tr_o)
                    best_params = (m, l, res_train["thresh"], res_train["direction"])
        if best_params:
            m, l, t, d = best_params
            # Filter test set for non-None
            t_data = [(s["trajectory"][l][m], target) for s, target in zip(X_te_o, y_te_o) if s["trajectory"][l][m] is not None]
            if t_data:
                y_pred = [1 if (v >= t if d == ">=" else v <= t) else 0 for v, _ in t_data]
                y_true_t = [t for _, t in t_data]
                scores.append(np.mean(np.array(y_pred) == np.array(y_true_t)))
    return np.mean(scores) if scores else 0.0

def compute_trajectory_features(traj, metric):
    vals = [t[metric] for t in traj if t[metric] is not None]
    if len(vals) < 5: return {"slope": 0.0, "ratio": 1.0}
    
    # 1. Slope (Central 75% to avoid model-init/un-embedding noise)
    start, end = int(len(vals)*0.1), int(len(vals)*0.85)
    window = vals[start:end]
    x = np.arange(len(window))
    slope = np.polyfit(x, window, 1)[0]
    
    # 2. Crossover Ratio (Early 25% / Late 25%)
    e_idx, l_idx = int(len(vals)*0.25), int(len(vals)*0.75)
    early = np.mean(vals[:e_idx]) if e_idx > 0 else vals[0]
    late = np.mean(vals[l_idx:]) if l_idx < len(vals) else vals[-1]
    ratio = early / late if late != 0 else 1.0
    
    return {"slope": slope, "ratio": ratio}

def run_mmt_analysis(results_data):
    """Multi-Metric Trajectory Logistic Regression."""
    X, y = [], []
    for label_key, label_val in [("radical", 1), ("neutral", 0)]:
        for s in results_data[label_key]:
            # Feature vector: [slope_fiedler, slope_hfer, slope_smooth, slope_entropy]
            features = []
            for m in ["fiedler_value", "hfer", "smoothness", "entropy"]:
                features.append(compute_trajectory_features(s["trajectory"], m)["slope"])
            X.append(features); y.append(label_val)
    
    X, y = np.array(X), np.array(y)
    if len(X) < 10: return {"acc": 0.0, "auroc": 0.0} # Not enough data for 5-fold CV
    
    pipeline = Pipeline([('scaler', StandardScaler()), ('lr', LogisticRegression())])
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    accs = cross_val_score(pipeline, X, y, cv=skf, scoring='accuracy')
    aurocs = cross_val_score(pipeline, X, y, cv=skf, scoring='roc_auc')
    
    return {"acc": np.mean(accs), "auroc": np.mean(aurocs)}

def compute_abc(data, metric):
    """Area Between Curves for radical vs neutral."""
    r_v = np.nanmean([[(t[metric] if t[metric] is not None else np.nan) for t in s["trajectory"]] for s in data["radical"]], axis=0)
    n_v = np.nanmean([[(t[metric] if t[metric] is not None else np.nan) for t in s["trajectory"]] for s in data["neutral"]], axis=0)
    return np.nansum(np.abs(r_v - n_v))

def print_summary_stats(results_data):
    print("\n" + "="*95)
    print(" SPECTRAL GEOMETRY OF EXTREMISM SUMMARY")
    print(f"Radical: {len(results_data['radical'])}, Neutral: {len(results_data['neutral'])}")
    if "metadata" in results_data:
        print(f"Nested CV Accuracy (MHS Gold Standard): {results_data['metadata']['nested_cv_accuracy']:.1%}")
        if "mmt" in results_data["metadata"]:
            print(f"MMT Logistic Regression Accuracy: {results_data['metadata']['mmt']['acc']:.1%}")
            print(f"MMT Logistic Regression AUROC: {results_data['metadata']['mmt']['auroc']:.2f}")
        
        best = results_data["metadata"]["best_metric"]
        if best:
            print(f"BEST SINGLE LAYER: {best['metric']} @ L{best['layer']} | d={best['d']:.2f} | AUROC={best['auroc']:.2f}")
        
        print("\nTrajectory Diagnostics:")
        for m in ["hfer", "smoothness"]:
            abc = results_data["metadata"].get("abc", {}).get(m, 0)
            print(f"  {m.upper()} Area Under Departure (ABC): {abc:.2f}")
    print("="*95 + "\n")

def perform_analysis(results_data, args):
    metrics = ["fiedler_value", "hfer", "smoothness", "entropy"]
    n_layers = len(results_data["radical"][0]["trajectory"]) if results_data["radical"] else 0
    all_res = []
    
    for l in range(n_layers):
        for m in metrics:
            r_v = [s["trajectory"][l][m] for s in results_data["radical"] if s["trajectory"][l][m] is not None]
            n_v = [s["trajectory"][l][m] for s in results_data["neutral"] if s["trajectory"][l][m] is not None]
            res = compute_t_stats(r_v, n_v)
            if res:
                res["layer"], res["metric"] = l, m
                all_res.append(res)
                
    all_res.sort(key=lambda x: abs(x["d"]), reverse=True)
    abc_stats = {m: compute_abc(results_data, m) for m in metrics}
    mmt_stats = run_mmt_analysis(results_data)
    
    results_data["metadata"] = {
        "model": args.model,
        "best_metric": all_res[0] if all_res else None,
        "top_10": all_res[:10],
        "nested_cv_accuracy": float(compute_nested_cv(results_data)),
        "abc": abc_stats,
        "mmt": mmt_stats
    }
    
    results_file = args.results_file or f"results/spectra/extremism_results_{args.model.split('/')[-1]}.json"
    with open(results_file, "w") as f:
        json.dump(results_data, f, indent=2)
    
    print_summary_stats(results_data)
    generate_all_plots(results_data, args)

def run_experiment(args):
    model_slug = args.model.split("/")[-1]
    results_file = args.results_file or f"results/spectra/extremism_results_{model_slug}.json"
    os.makedirs(os.path.dirname(results_file), exist_ok=True)
    
    if args.plot_only:
        if not os.path.exists(results_file):
            print(f"Error: Results file {results_file} not found for --plot-only.")
            return
        with open(results_file, "r") as f:
            results_data = json.load(f)
        perform_analysis(results_data, args)
        return

    model_kwargs = {}
    model_kwargs["use_cache"] = False
    model_kwargs["attn_implementation"] = "eager"
    model_kwargs["output_attentions"] = True
    model_kwargs["output_hidden_states"] = True
    if args.load_in_8bit or args.load_in_4bit:
        from transformers import BitsAndBytesConfig
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_8bit=args.load_in_8bit,
            load_in_4bit=args.load_in_4bit,
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16,
            bnb_4bit_quant_type="nf4"
        )
    
    config = GSPConfig(model_name=args.model, device=args.device, 
                       local_files_only=args.offline, 
                       trust_remote_code=True,
                       model_kwargs=model_kwargs)
    
    results_data = {"radical": [], "neutral": []}
    with open(args.dataset, "r") as f: dataset = json.load(f)
    
    with GSPDiagnosticsFramework(config) as framework:
        framework.instrumenter.load_model(args.model)
        
        for item in tqdm(dataset, desc=f"Extracting {model_slug}"):
            try:
                analysis = framework.analyze_text(item["text"], save_results=False)
                traj = [{"layer": i, "fiedler_value": float(d.fiedler_value) if d.fiedler_value else None,
                         "hfer": float(d.hfer) if d.hfer else None, "smoothness": float(d.smoothness_index) if d.smoothness_index else None,
                         "entropy": float(d.spectral_entropy) if d.spectral_entropy else None} 
                        for i, d in enumerate(analysis['layer_diagnostics'])]
                results_data["radical" if item["label"] == 1 else "neutral"].append({
                    "id": item.get("id"), "category": item.get("category"), "trajectory": traj})
                
                del analysis
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception as e: 
                print(f"Error on item: {e}")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    perform_analysis(results_data, args)

def generate_all_plots(data, args):
    os.makedirs("results/figures", exist_ok=True)
    model_slug = args.model.split("/")[-1]
    metrics = ["fiedler_value", "hfer", "smoothness", "entropy"]
    fig, axes = plt.subplots(1, len(metrics), figsize=(24, 6))
    
    n_rad = len(data["radical"])
    n_neu = len(data["neutral"])
    
    for i, m in enumerate(metrics):
        for k, c, label in [("radical", "#e74c3c", f"Radical (N={n_rad})"), 
                            ("neutral", "#3498db", f"Neutral (N={n_neu})")]:
            if not data[k]: continue
            # Handle None values by converting to nan
            all_v = [[(t[m] if t[m] is not None else np.nan) for t in s["trajectory"]] for s in data[k]]
            all_v = np.array(all_v)
            mean_v = np.nanmean(all_v, axis=0)
            std_v = np.nanstd(all_v, axis=0)
            
            layers = np.arange(len(mean_v))
            axes[i].plot(layers, mean_v, color=c, linewidth=3, label=label, alpha=0.9)
            axes[i].fill_between(layers, mean_v - std_v, mean_v + std_v, color=c, alpha=0.15)
            
        axes[i].set_title(f"Spectral {m.capitalize()}", fontsize=14, fontweight='bold')
        axes[i].set_xlabel("Layer Index", fontsize=12)
        axes[i].set_ylabel("Value", fontsize=12)
        axes[i].grid(True, linestyle='--', alpha=0.6)
        axes[i].legend(loc="best", frameon=True, fontsize=10)
        
    plt.suptitle(f"Spectral Trajectories: {args.model}", fontsize=18, y=1.02, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"results/figures/trajectories_{model_slug}.png", dpi=300, bbox_inches='tight')
    print(f"Premium trajectory plot saved to results/figures/trajectories_{model_slug}.png")

if __name__ == "__main__":
    args = parse_args(); run_experiment(args)
