import os
import json
import argparse
import numpy as np
import pandas as pd
from scipy import stats
from scipy.interpolate import interp1d
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def compute_t_stats(radical, neutral):
    """Exhaustive stats: Cohen's d, AUROC, Mann-Whitney p."""
    if len(radical) < 2 or len(neutral) < 2: return {"d": 0, "auroc": 0.5, "p": 1.0}
    mu_r, std_r = np.mean(radical), np.std(radical)
    mu_n, std_n = np.mean(neutral), np.std(neutral)
    try: _, p_mw = stats.mannwhitneyu(radical, neutral, alternative='two-sided')
    except: p_mw = 1.0
    pooled_std = np.sqrt(((len(radical)-1)*std_r**2 + (len(neutral)-1)*std_n**2) / (len(radical)+len(neutral)-2))
    d = (mu_r - mu_n) / (pooled_std + 1e-9)
    y_true = [1]*len(radical) + [0]*len(neutral)
    scores = list(radical) + list(neutral)
    try: auroc = roc_auc_score(y_true, scores)
    except: auroc = 0.5
    return {"d": d, "auroc": auroc, "p": p_mw}

def analyze_model_features(results_path, quiet=False):
    if not quiet: print(f"\n>>> Analyzing Advanced Features for: {os.path.basename(results_path)}")
    with open(results_path, "r") as f: data = json.load(f)
    
    metrics = ["fiedler_value", "hfer", "smoothness", "entropy"]
    radical_trajs = [s["trajectory"] for s in data["radical"]]
    all_samples = data["radical"] + data["neutral"]
    y = np.array([1]*len(data["radical"]) + [0]*len(data["neutral"]))
    n_layers = len(radical_trajs[0]) if radical_trajs else 0

    X_raw = []
    for s in all_samples:
        traj = s["trajectory"]
        raw_vec = []
        for m in metrics:
            v_m = np.array([(t[m] if t[m] is not None else np.nan) for t in traj])
            mask = ~np.isnan(v_m)
            if np.any(mask):
                v_m[np.isnan(v_m)] = np.interp(np.flatnonzero(np.isnan(v_m)), np.flatnonzero(mask), v_m[mask])
            else: v_m[:] = 0
            raw_vec.extend(v_m)
        X_raw.append(raw_vec)
    X_raw = np.array(X_raw)

    results = {}
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    def eval_model(name, X, model):
        pipeline = Pipeline([('scaler', StandardScaler()), ('clf', model)])
        aurocs = cross_val_score(pipeline, X, y, cv=skf, scoring='roc_auc')
        results[name] = np.mean(aurocs)

    eval_model("LogReg", X_raw, LogisticRegression(max_iter=1000, C=1.0))
    eval_model("SVM", X_raw, SVC(probability=True, kernel='linear')) # Use linear SVM for comparable baseline

    # Baseline hand-crafted estimation (HFER Slope)
    v_hfer = X_raw[:, n_layers:2*n_layers]
    slopes = [stats.linregress(range(n_layers), v_hfer[i]).slope for i in range(len(v_hfer))]
    stats_res = compute_t_stats(np.array(slopes)[:len(data["radical"])], np.array(slopes)[len(data["radical"]):])
    results["Hand-crafted"] = stats_res["auroc"]

    if not quiet:
        print(f"| LogReg AUROC: {results['LogReg']:.3f} | SVM AUROC: {results['SVM']:.3f} | Hand-crafted AUROC: {results['Hand-crafted']:.3f} |")
    
    return results

def print_mechanistic_interpretation(results_path):
    print(f"\n>>> MECHANISTIC INTERPRETATION (Llama-3.2-3B)")
    with open(results_path, "r") as f: data = json.load(f)
    metrics = ["fiedler_value", "hfer", "smoothness", "entropy"]
    all_samples = data["radical"] + data["neutral"]
    y = np.array([1]*len(data["radical"]) + [0]*len(data["neutral"]))
    n_layers = len(all_samples[0]["trajectory"])

    X_raw = []
    for s in all_samples:
        traj = s["trajectory"]
        raw_vec = []
        for m in metrics:
            v_m = np.array([(t[m] if t[m] is not None else np.nan) for t in traj])
            mask = ~np.isnan(v_m)
            if np.any(mask):
                v_m[np.isnan(v_m)] = np.interp(np.flatnonzero(np.isnan(v_m)), np.flatnonzero(mask), v_m[mask])
            else: v_m[:] = 0
            raw_vec.extend(v_m)
        X_raw.append(raw_vec)
    X_raw = np.array(X_raw)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)
    model = LogisticRegression(max_iter=1000, C=1.0)
    model.fit(X_scaled, y)
    
    coeffs = model.coef_[0]
    feature_labels = []
    for m in metrics:
        for l in range(n_layers):
            feature_labels.append(f"{m}_L{l}")
    
    df_coeffs = pd.DataFrame({"Feature": feature_labels, "Weight": coeffs})
    df_coeffs = df_coeffs.sort_values("Weight", ascending=False)
    
    print("\nTOP 10 PRO-RADICAL INDICATORS (Positive Weights):")
    print(df_coeffs.head(10).to_markdown(index=False))
    
    print("\nTOP 10 PRO-NEUTRAL INDICATORS (Negative Weights):")
    print(df_coeffs.tail(10).sort_values("Weight").to_markdown(index=False))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", help="Path to results JSON")
    parser.add_argument("--interpret", action="store_true", help="Print feature coefficients")
    args = parser.parse_args()
    
    if args.interpret:
        print_mechanistic_interpretation(args.results)
    elif args.results:
        analyze_model_features(args.results)
