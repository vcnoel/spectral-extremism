# Spectral Geometry of Extremism: Implementation Guide

## For: Coding Assistant / Research Engineer
## Goal: Replicate the "Geometry of Reason" study design for radicalized text detection
## Library: `spectral-trust` (our own)

---

## What We're Doing

Paper: "Geometry of Reason" showed that spectral analysis of attention graphs separates valid from invalid math proofs (Cohen's d up to 3.30, 85–96% accuracy, training-free). We replicate **that exact study design** — same metrics, same statistical tests, same classification approach — but on a different binary task: **radicalized text vs. neutral text**.

The pipeline is:

```
Geometry of Reason:  valid proof vs. invalid proof  → spectral diagnostics → separation?
Our experiment:      radical text vs. neutral text   → spectral diagnostics → separation?
```

Same method, new data. That's it.

---

## STEP ZERO: Setup

### 1. Get the Geometry of Reason codebase as reference

```bash
git clone https://anonymous.4open.science/r/geometry-of-reason-EF4B/
# or: git clone https://github.com/vnoël/geometry-of-reason.git
```

**This is your reference, not your dependency.** Read it to understand the experiment structure, the statistical reporting, and the analysis pipeline. You will replicate this structure in your own code.

### 2. Install spectral-trust

`spectral-trust` is our library. It already implements the full spectral pipeline (attention extraction, symmetrization, Laplacian, eigendecomposition, all 4 diagnostics). Install it:

```bash
pip install spectral-trust
# or if it's a local package:
pip install -e /path/to/spectral-trust
```

**Read the spectral-trust API first.** Find the main functions:
- How to load a model with attention extraction enabled
- How to run a forward pass and get spectral diagnostics per layer
- What the output format is (dict? dataclass? DataFrame?)

All spectral computation goes through `spectral-trust`. Do not reimplement anything from Geometry of Reason's spectral code.

### 3. Other dependencies

```bash
pip install torch transformers numpy scipy pandas matplotlib seaborn tqdm datasets --break-system-packages
```

---

## Part 1: Models

Same models as Geometry of Reason. They span 4 architectural families:

```python
MODELS = [
    "meta-llama/Llama-3.2-1B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
    "meta-llama/Llama-3.1-8B-Instruct",
    "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "microsoft/Phi-3.5-mini-instruct",
    "mistralai/Mistral-7B-Instruct-v0.1",
]
```

**Start with Llama-3.2-3B-Instruct only.** Get the full pipeline working end-to-end on one model before scaling.

---

## Part 2: Dataset

### 2.1 Structure: Mirror Geometry of Reason's MiniF2F

Geometry of Reason used the MiniF2F benchmark: ~454 samples, ~40% valid, ~60% invalid, each sample is a text string with a binary label.

We need the same thing: ~400–500 text samples, ~40% radical, ~60% neutral, each sample is a text string with a binary label.

```python
# Target dataset structure — identical to how Geometry of Reason structures MiniF2F
dataset = [
    {"id": "rad_001", "text": "The invasion of our homeland...", "label": 1, "category": "immig"},
    {"id": "neu_001", "text": "Immigration policy requires...", "label": 0, "category": "immig"},
    ...
]
# label: 1 = radical, 0 = neutral
```

### 2.2 Sources

**Radical texts (target: ~180–200 samples):**

Use established, labeled corpora. Priority order:

1. **Jigsaw Toxic Comments** (Kaggle): filter for `severe_toxic` or `identity_hate` labels. Plenty of data, well-labeled, English.
   ```bash
   # Download from: https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge
   ```

2. **Gab Hate Corpus** (Kennedy et al., 2020): annotated Gab posts. Use posts labeled as hate speech with ideological framing.

3. **Implicit Hate Corpus** (ElSherief et al., 2021): specifically captures implicit/coded hate speech — exactly the kind of content we hope spectral methods can detect.

4. **Stormfront dataset** (de Gibert et al., 2018): annotated forum posts from a white supremacist forum.

5. **Manual curation**: 20–30 samples of known extremist rhetoric from public research reports (EU RAN, GPAHE, SPLC). These serve as "gold standard" radical texts.

**Neutral texts (target: ~250–300 samples):**

Topically matched. For every radical text about immigration, include a neutral text about immigration.

- Wikipedia paragraphs on matching topics
- Reuters/AP news articles on matching topics
- Academic abstracts from social science papers on matching topics

### 2.3 Category Labels (Secondary, for Analysis)

Every sample gets a primary label (`radical` / `neutral`) and a secondary category:

| Category | Code | Examples |
|---|---|---|
| Anti-immigration | `immig` | Replacement rhetoric vs. policy discussion |
| Violent extremism | `violent` | Martyrdom glorification vs. conflict reporting |
| Conspiracy | `conspir` | Cabal narratives vs. institutional critique |
| Ethnonationalism | `ethno` | Racial purity vs. cultural heritage discussion |
| Polarization | `polar` | Us-vs-them absolutism vs. partisan debate |
| Generic neutral | `generic` | Neutral texts not matched to a specific radical category |

### 2.4 Controls (CRITICAL)

Add these as separate labeled conditions in the same dataset:

**Control A — Toxic but not ideological (~50 samples, label: 0)**
Personal insults, profanity, aggressive language with no political/ideological content.
```
"You're a complete idiot and your argument makes zero sense. Shut up."
```
Label as `neutral` (label=0) with `category="toxic_control"`.

**Control B — Partisan but not radical (~50 samples, label: 0)**
Strong political opinions within normal democratic discourse.
```
"Corporate greed is destroying the middle class. We need radical economic reform now."
```
Label as `neutral` (label=0) with `category="partisan_control"`.

These controls test whether we detect *radicalization specifically* vs. *negativity* or *strong opinions*.

### 2.5 Length Matching

**Check and enforce token-length compatibility.** Geometry of Reason showed accuracy was stable across proof length quintiles (Table 11 in that paper). We need to verify the same, but first, avoid systematic length confounds:

```python
def filter_by_length(samples, tokenizer, min_tokens=20, max_tokens=200):
    """Keep only samples within a reasonable token range."""
    filtered = []
    for s in samples:
        n_tokens = len(tokenizer.encode(s["text"]))
        if min_tokens <= n_tokens <= max_tokens:
            s["n_tokens"] = n_tokens
            filtered.append(s)
    return filtered
```

### 2.6 Save the Dataset

```python
import json

# Save as JSON — same simplicity as MiniF2F
with open("data/extremism_dataset.json", "w") as f:
    json.dump(dataset, f, indent=2)

# Print stats
n_radical = sum(1 for s in dataset if s["label"] == 1)
n_neutral = sum(1 for s in dataset if s["label"] == 0)
print(f"Dataset: {len(dataset)} samples, {n_radical} radical ({n_radical/len(dataset)*100:.1f}%), {n_neutral} neutral")
# Target: ~40/60 split, matching Geometry of Reason's MiniF2F balance
```

---

## Part 3: Running Spectral Extraction

### 3.1 Use spectral-trust

Call `spectral-trust` for all the spectral computation. The loop mirrors Geometry of Reason's pipeline:

```python
import spectral_trust  # OUR LIBRARY — check the actual API and adapt the calls below
import torch
from tqdm import tqdm
import json

def run_extraction(model_name, dataset_path, output_path):
    """
    For every text in the dataset, run a forward pass and extract
    spectral diagnostics at every layer.
    
    This mirrors Geometry of Reason Section 3.4:
    1. Pass text through transformer
    2. Extract attention matrices and hidden states
    3. Compute spectral diagnostics at each layer
    """
    
    # Load model via spectral-trust (check the actual API)
    # Something like:
    model, tokenizer = spectral_trust.load_model(model_name)
    # or:
    # analyzer = spectral_trust.SpectralAnalyzer(model_name)
    
    with open(dataset_path) as f:
        dataset = json.load(f)
    
    results = []
    
    for sample in tqdm(dataset, desc=model_name):
        text = sample["text"]
        
        # Extract spectral profile via spectral-trust
        # Check the actual API — it should return per-layer diagnostics
        # Something like:
        profile = spectral_trust.extract_profile(model, tokenizer, text)
        # or:
        # profile = analyzer.analyze(text)
        #
        # profile should be a list of dicts, one per layer:
        # [{"layer": 0, "fiedler": 0.45, "hfer": 0.12, "smoothness": 0.87, "spectral_entropy": 1.23}, ...]
        
        results.append({
            "id": sample["id"],
            "label": sample["label"],          # 1=radical, 0=neutral
            "category": sample["category"],
            "n_tokens": len(tokenizer.encode(text)),
            "spectral_profile": profile,
        })
    
    # Save results
    with open(output_path, "w") as f:
        json.dump(results, f, default=str)
    
    print(f"Saved {len(results)} results to {output_path}")
    return results
```

### 3.2 Run

```bash
# Start with one model, verify it works
python main.py extract --model meta-llama/Llama-3.2-3B-Instruct --dataset data/extremism_dataset.json

# Then run analysis on that output
python main.py analyze --results results/spectra/Llama-3.2-3B-Instruct.json

# Or do everything at once
python main.py all --model meta-llama/Llama-3.2-3B-Instruct --dataset data/extremism_dataset.json

# Scale to all models
for model in meta-llama/Llama-3.2-1B-Instruct meta-llama/Llama-3.2-3B-Instruct meta-llama/Llama-3.1-8B-Instruct Qwen/Qwen2.5-0.5B-Instruct Qwen/Qwen2.5-7B-Instruct microsoft/Phi-3.5-mini-instruct mistralai/Mistral-7B-Instruct-v0.1; do
    python main.py extract --model $model --dataset data/extremism_dataset.json
    python main.py analyze --results results/spectra/$(basename $model).json
done
```

---

## Part 4: Analysis — Mirror Geometry of Reason Exactly

### 4.1 Main Results Table (Replicate Table 2)

Geometry of Reason's Table 2 reports, per model: best metric, best layer, pMW, pt, |d|, accuracy.

Reproduce this EXACTLY for our data:

```python
import numpy as np
from scipy.stats import mannwhitneyu, ttest_ind
import pandas as pd

def compute_main_results(results):
    """
    Replicate Geometry of Reason Table 2.
    For each (metric, layer), compute effect size and significance.
    Report the best combination.
    """
    metrics = ["fiedler", "hfer", "smoothness", "spectral_entropy"]
    num_layers = len(results[0]["spectral_profile"])
    
    radical = [r for r in results if r["label"] == 1]
    neutral = [r for r in results if r["label"] == 0]
    
    rows = []
    
    for layer_idx in range(num_layers):
        for metric in metrics:
            rad_vals = np.array([r["spectral_profile"][layer_idx][metric] for r in radical])
            neu_vals = np.array([r["spectral_profile"][layer_idx][metric] for r in neutral])
            
            # Cohen's d (matching Geometry of Reason: d = (μ_A - μ_B) / s_pooled)
            n_r, n_n = len(rad_vals), len(neu_vals)
            pooled_std = np.sqrt(
                ((n_r - 1) * rad_vals.std(ddof=1)**2 + (n_n - 1) * neu_vals.std(ddof=1)**2)
                / (n_r + n_n - 2)
            )
            d = (rad_vals.mean() - neu_vals.mean()) / pooled_std if pooled_std > 0 else 0.0
            
            # Mann-Whitney U (pMW)
            stat_mw, p_mw = mannwhitneyu(rad_vals, neu_vals, alternative="two-sided")
            
            # Welch's t-test (pt)
            stat_t, p_t = ttest_ind(rad_vals, neu_vals, equal_var=False)
            
            # Single-threshold accuracy (Section 3.4 of Geometry of Reason)
            all_vals = np.concatenate([rad_vals, neu_vals])
            all_labels = np.array([1]*n_r + [0]*n_n)
            
            best_acc = 0
            best_thresh = 0
            for thresh in np.percentile(all_vals, np.arange(1, 100)):
                acc_pos = ((all_vals >= thresh) == all_labels).mean()
                acc_neg = ((all_vals < thresh) == all_labels).mean()
                acc = max(acc_pos, acc_neg)
                if acc > best_acc:
                    best_acc = acc
                    best_thresh = thresh
            
            rows.append({
                "metric": metric,
                "layer": layer_idx,
                "d": d,
                "abs_d": abs(d),
                "p_mw": p_mw,
                "p_t": p_t,
                "accuracy": best_acc,
                "radical_mean": rad_vals.mean(),
                "neutral_mean": neu_vals.mean(),
                "threshold": best_thresh,
            })
    
    df = pd.DataFrame(rows)
    
    # Report best per model (matching Table 2 format)
    best = df.loc[df["abs_d"].idxmax()]
    print(f"\n=== BEST DISCRIMINATOR ===")
    print(f"Metric: {best['metric']} @ Layer {best['layer']}")
    print(f"|d| = {best['abs_d']:.2f}")
    print(f"pMW = {best['p_mw']:.2e}")
    print(f"pt = {best['p_t']:.2e}")
    print(f"Accuracy = {best['accuracy']:.1%}")
    print(f"Radical mean: {best['radical_mean']:.4f}")
    print(f"Neutral mean: {best['neutral_mean']:.4f}")
    
    return df
```

### 4.2 Top-10 Discriminators Table (Replicate Tables 19–23)

Geometry of Reason reports the top 10 (metric, layer) combinations per model. Do the same:

```python
def top_discriminators(df, n=10):
    """Replicate Geometry of Reason Appendix F tables."""
    top = df.nlargest(n, "abs_d")
    print(top[["metric", "layer", "d", "p_mw", "p_t", "accuracy",
               "radical_mean", "neutral_mean"]].to_string())
    return top
```

### 4.3 Nested Cross-Validation (Replicate Table 7)

This is the rigorous evaluation. Geometry of Reason used 5-fold outer / 4-fold inner CV to select (metric, layer, threshold) without leakage:

```python
from sklearn.model_selection import StratifiedKFold

def nested_cv(results, outer_k=5, inner_k=4):
    """
    Replicate Geometry of Reason Section 4.2 nested CV.
    Outer loop: evaluate generalization.
    Inner loop: select best (metric, layer, threshold).
    """
    labels = np.array([r["label"] for r in results])
    num_layers = len(results[0]["spectral_profile"])
    metrics = ["fiedler", "hfer", "smoothness", "spectral_entropy"]
    
    outer_cv = StratifiedKFold(n_splits=outer_k, shuffle=True, random_state=42)
    outer_accs = []
    best_configs = []
    
    for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(np.zeros(len(labels)), labels)):
        train_results = [results[i] for i in train_idx]
        test_results = [results[i] for i in test_idx]
        train_labels = labels[train_idx]
        test_labels = labels[test_idx]
        
        # Inner CV: select best (metric, layer, threshold, direction)
        inner_cv = StratifiedKFold(n_splits=inner_k, shuffle=True, random_state=123)
        best_inner_score = 0
        best_config = None
        
        for metric in metrics:
            for layer_idx in range(num_layers):
                fold_accs = []
                
                for itrain, ival in inner_cv.split(np.zeros(len(train_labels)), train_labels):
                    itrain_vals = np.array([train_results[i]["spectral_profile"][layer_idx][metric] for i in itrain])
                    itrain_labs = train_labels[itrain]
                    ival_vals = np.array([train_results[i]["spectral_profile"][layer_idx][metric] for i in ival])
                    ival_labs = train_labels[ival]
                    
                    # Find best threshold + direction on inner train
                    best_fold_acc = 0
                    for thresh in np.percentile(itrain_vals, np.arange(5, 96, 5)):
                        for direction in [1, -1]:
                            preds = (direction * ival_vals >= direction * thresh).astype(int)
                            acc = (preds == ival_labs).mean()
                            if acc > best_fold_acc:
                                best_fold_acc = acc
                    fold_accs.append(best_fold_acc)
                
                mean_acc = np.mean(fold_accs)
                if mean_acc > best_inner_score:
                    best_inner_score = mean_acc
                    best_config = (metric, layer_idx)
        
        # Evaluate on outer test with selected config
        metric, layer_idx = best_config
        train_vals = np.array([r["spectral_profile"][layer_idx][metric] for r in train_results])
        test_vals = np.array([r["spectral_profile"][layer_idx][metric] for r in test_results])
        
        # Calibrate threshold on full training set
        best_test_acc = 0
        for thresh in np.percentile(train_vals, np.arange(5, 96, 5)):
            for direction in [1, -1]:
                preds = (direction * test_vals >= direction * thresh).astype(int)
                acc = (preds == test_labels).mean()
                if acc > best_test_acc:
                    best_test_acc = acc
        
        outer_accs.append(best_test_acc)
        best_configs.append(best_config)
        print(f"  Fold {fold_idx+1}: {best_test_acc:.1%} (config: {best_config[0]}@L{best_config[1]})")
    
    print(f"\nNested CV Accuracy: {np.mean(outer_accs):.1%} ± {np.std(outer_accs):.1%}")
    print(f"Most selected config: {max(set(best_configs), key=best_configs.count)}")
    return outer_accs
```

### 4.4 Ablations (Replicate Section 4.4)

Run these SAME ablations from Geometry of Reason:

**1. Random baseline.** Majority class accuracy (always predict "neutral") = ~60%. Our method should beat this substantially.

**2. Threshold robustness.** Perturb optimal threshold by ±10%, ±20%. Report accuracy drop. (Geometry of Reason saw <2.5% drop at ±10%.)

**3. Text length.** Stratify by token-count quintiles. Report accuracy per quintile. (Must be stable — otherwise we're just measuring length.)

**4. Category breakdown.** Stratify by ideology category. Report accuracy per category. (Analog of Geometry of Reason's "problem difficulty" stratification, Table 10.)

**5. Metric correlations.** Compute pairwise Pearson correlations between the 4 metrics at the best layer. (Geometry of Reason found HFER and Entropy nearly redundant, r = −0.97, while Fiedler was independent.)

**6. Multiple comparisons.** Apply Benjamini-Hochberg correction at FDR = 0.05 over all (metric, layer) hypotheses. Report how many survive.

### 4.5 Control Analysis (NEW — Not in Geometry of Reason)

This is the only analysis that doesn't have a direct analog in Geometry of Reason. Run the same separation analysis on subsets:

```python
def run_control_comparisons(results):
    """
    The critical comparisons that determine whether we detect
    radicalization specifically or just negativity/strong opinions.
    """
    radical = [r for r in results if r["label"] == 1]
    neutral_clean = [r for r in results if r["label"] == 0 and r["category"] not in ("toxic_control", "partisan_control")]
    toxic = [r for r in results if r["category"] == "toxic_control"]
    partisan = [r for r in results if r["category"] == "partisan_control"]
    
    comparisons = [
        ("Radical vs. Neutral", radical, neutral_clean),
        ("Radical vs. Partisan", radical, partisan),
        ("Radical vs. Toxic", radical, toxic),
        ("Toxic vs. Neutral", toxic, neutral_clean),
        ("Partisan vs. Neutral", partisan, neutral_clean),
    ]
    
    for name, group_a, group_b in comparisons:
        if len(group_a) < 10 or len(group_b) < 10:
            print(f"{name}: SKIPPED (insufficient samples)")
            continue
        # Reuse compute_main_results with these two groups
        # Find best |d| across all (metric, layer)
        # Report it
        print(f"{name}: best |d| = ?, pMW = ?")
```

Present as a table:

| Comparison | Best |d| | p_MW | Interpretation |
|---|---|---|---|
| Radical vs. Neutral | ? | ? | Primary effect |
| Radical vs. Partisan | ? | ? | Radicalization beyond strong opinions? |
| Radical vs. Toxic | ? | ? | Ideology beyond toxicity? |
| Toxic vs. Neutral | ? | ? | Toxicity spectral baseline |
| Partisan vs. Neutral | ? | ? | Opinion-strength spectral baseline |

**The key result:** if |d|(radical vs. neutral) >> |d|(partisan vs. neutral), we're detecting something specific to radical rhetoric, not just opinion strength.

### 4.6 Layer-wise Profile Plots (Replicate Figures 6–13)

Geometry of Reason plots all 4 metrics across layers with valid (blue) and invalid (red) bands. Replicate exactly, substituting:
- Blue = neutral
- Red = radical

One 4-panel figure per model. Same matplotlib style as the paper. Copy plotting code from the Geometry of Reason repo if available.

---

## Part 5: Architecture — ONE CLI, Clean Code

### DO NOT create a bunch of separate scripts. Build ONE CLI with subcommands.

```
spectral-extremism/
├── data/
│   └── extremism_dataset.json
├── results/
│   ├── spectra/                        # Per-model JSON outputs from `extract`
│   ├── tables/                         # CSVs from `analyze`
│   └── figures/                        # PNGs from `analyze`
├── main.py                             # THE SINGLE ENTRY POINT
└── README.md
```

### `main.py` — One file, subcommands via argparse:

```python
"""
Spectral Geometry of Extremism.
Replicates Geometry of Reason study design for radicalized text detection.
All spectral computation via spectral-trust.

Usage:
    python main.py extract  --model meta-llama/Llama-3.2-3B-Instruct --dataset data/extremism_dataset.json
    python main.py analyze  --results results/spectra/llama3b.json
    python main.py ablations --results results/spectra/llama3b.json
    python main.py controls --results results/spectra/llama3b.json
    python main.py plot     --results results/spectra/llama3b.json
    python main.py all      --model meta-llama/Llama-3.2-3B-Instruct --dataset data/extremism_dataset.json
"""

import argparse

def cmd_extract(args):
    """Part 3: Run spectral-trust on every sample, save per-layer profiles."""
    ...

def cmd_analyze(args):
    """Part 4.1–4.3: Main results table, top-10 discriminators, nested CV."""
    ...

def cmd_ablations(args):
    """Part 4.4: Threshold robustness, length stratification, metric correlations, BH correction."""
    ...

def cmd_controls(args):
    """Part 4.5: Radical vs. partisan vs. toxic comparisons."""
    ...

def cmd_plot(args):
    """Part 4.6: Layer-wise 4-panel profile figures."""
    ...

def cmd_all(args):
    """Run everything end-to-end."""
    cmd_extract(args)
    cmd_analyze(args)
    cmd_ablations(args)
    cmd_controls(args)
    cmd_plot(args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Spectral Geometry of Extremism")
    sub = parser.add_subparsers(dest="command")

    p_ext = sub.add_parser("extract")
    p_ext.add_argument("--model", required=True)
    p_ext.add_argument("--dataset", required=True)
    p_ext.add_argument("--output", default="results/spectra/")

    p_ana = sub.add_parser("analyze")
    p_ana.add_argument("--results", required=True)

    p_abl = sub.add_parser("ablations")
    p_abl.add_argument("--results", required=True)

    p_ctrl = sub.add_parser("controls")
    p_ctrl.add_argument("--results", required=True)

    p_plot = sub.add_parser("plot")
    p_plot.add_argument("--results", required=True)
    p_plot.add_argument("--output", default="results/figures/")

    p_all = sub.add_parser("all")
    p_all.add_argument("--model", required=True)
    p_all.add_argument("--dataset", required=True)

    args = parser.parse_args()
    globals()[f"cmd_{args.command}"](args)
```

### Rules for the coding assistant:

1. **Everything lives in `main.py`.** Helper functions go in the same file. If it grows past ~800 lines, you may split into `main.py` + `utils.py`. Not before.
2. **No Jupyter notebooks.** Everything runs from the command line.
3. **`extract` saves JSON. Everything else reads that JSON.** The extraction step is the expensive one (GPU). Analysis, ablations, controls, and plots are all cheap and re-runnable from saved results.
4. **`python main.py all`** runs the complete pipeline end-to-end. This must work.
5. **Every subcommand prints its results to stdout AND saves to a file.** Tables go to `results/tables/`, figures go to `results/figures/`.

---

## Part 6: Execution Order

| Day | Task | Command |
|-----|------|---------|
| 1 | Build dataset. Save as `data/extremism_dataset.json`. | Manual / script |
| 2 | Implement `cmd_extract` in `main.py`. Test on 10 samples. | `python main.py extract --model meta-llama/Llama-3.2-3B-Instruct --dataset data/extremism_dataset.json` |
| 3 | Run Llama-3B full extraction. Implement + run `cmd_analyze`. | `python main.py analyze --results results/spectra/llama3b.json` |
| 4 | If signal: implement `cmd_ablations`, `cmd_plot`, `cmd_controls`. If no signal: debug. | `python main.py all --model meta-llama/Llama-3.2-3B-Instruct --dataset data/extremism_dataset.json` |
| 5 | Run remaining models. | Same `extract` + `analyze` per model |
| 6 | Run control comparisons across all models. | `python main.py controls --results results/spectra/*.json` |
| 7 | Compile results. Report back. | |

---

## Part 7: What To Report Back

1. **Table 2 analog**: per model, best (metric, layer), |d|, pMW, pt, accuracy.
2. **Nested CV accuracy**: per model, mean ± std across 5 folds.
3. **Layer-wise profile plots**: 4-panel figures, radical (red) vs. neutral (blue), per model.
4. **Control comparison table**: does radical separate from partisan and toxic?
5. **Category breakdown**: which types of extremism produce the strongest signal?
6. **Ablation results**: threshold robustness, length invariance, metric correlations.
7. **Any surprises.**

Do NOT spend time on: modifying spectral-trust, trying different Laplacians, training classifiers, directed analysis, or multilingual extension.

---

## Part 8: What Success Looks Like

| Result | Assessment |
|---|---|
| |d| > 2.0, nested CV > 80% | Geometry-of-Reason-level result. Strong publication. |
| |d| = 1.0–2.0, nested CV > 70% | Strong signal. Publishable with clean controls. |
| |d| = 0.5–1.0, nested CV > 65% | Moderate signal. Publishable if radical >> partisan in controls. |
| |d| < 0.5 | Weak or null. Check if toxic/partisan also ~0.3–0.5. Publishable as nuanced negative result. |

Realistic expectation: d = 0.5–1.5. Radical text is messier than math proofs, but the controls determine whether the signal is meaningful.

---

## Part 9: Pitfalls

1. **Topic confound.** If radical texts are all about violence and neutral texts are about cooking, you measure topic not ideology. Every radical text needs a topically matched neutral counterpart.

2. **Length confound.** Filter to 20–200 tokens. Report accuracy stratified by length quintiles.

3. **Chat template / safety circuitry.** Instruct models may activate refusal circuits on radical text, changing attention patterns. Try both with and without chat template. Try base models alongside instruct. If instruct models show a stronger signal than base models, part of the signal is the model's safety training — report this explicitly.

4. **Small N for controls.** 50 toxic + 50 partisan controls may be underpowered for detecting small differences. If control comparisons are inconclusive, flag this and plan to expand in follow-up.

5. **Label noise.** Jigsaw labels are crowd-sourced and noisy. Manually verify a random 10% sample of your dataset before running the full experiment.
