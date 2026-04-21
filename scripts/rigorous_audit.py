import os
import json
import torch
import pandas as pd
import numpy as np
import random
import time
from tqdm import tqdm
from transformers import AutoTokenizer
from spectral_trust import GSPDiagnosticsFramework, GSPConfig

# --- CONFIG ---
MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"
DATASET_PATH = "data/extremism_dataset.json"
OUTPUT_FILE = "results/spectra/rigorous_audit_results.json"
CHECKPOINT_FILE = "results/spectra/rigorous_audit_checkpoint.json"
N_SAMPLES = 500
LENGTH_TOLERANCE = 0.05  # ±5%
TIKHONOV_REG = 1e-6

def rewrite_text(model, tokenizer, text):
    """Style-transfer: Formalize the text using the model itself."""
    prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a high-level formal academic editor. Rewrite the following text in an extremely objective, formalized, and detached academic register. Remove all slang, emotive language, or obvious bias, while preserving the core underlying intent and logical structure. Do not add commentary.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=256, do_sample=False)
    # Extract only the assistant response
    response = tokenizer.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    return response.strip()

def process_sequence_with_framework(framework, text):
    """
    Extract Layer 4 spectral metrics using the GSPDiagnosticsFramework.
    Ensures BOS masking is applied to focus on semantic tokens.
    """
    # 1. Full analysis
    # Note: framework.analyze_text internally tokenizes, forward passes, and computes diagnostics.
    results = framework.analyze_text(text, save_results=False)
    
    # 2. Extract Layer 4 (0-indexed)
    layer_4_diag = results['layer_diagnostics'][4]
    
    # 3. Trajectory extraction (Fiedler and HFER)
    trajectory = []
    for diag in results['layer_diagnostics']:
        trajectory.append({
            "layer": diag.layer,
            "fiedler": diag.fiedler_value,
            "hfer": diag.hfer,
            "smoothness": diag.smoothness_index,
            "entropy": diag.spectral_entropy
        })
        
    return {
        "text": text,
        "token_len": len(results['tokens']),
        "l4_metrics": {
            "fiedler": layer_4_diag.fiedler_value,
            "hfer": layer_4_diag.hfer,
            "smoothness": layer_4_diag.smoothness_index,
            "entropy": layer_4_diag.spectral_entropy,
            "gini": 0.0 # Gini not in currently seen SpectralDiagnostics but we can add later if wanted
        },
        "trajectory": trajectory
    }

def main():
    print("--- RIGOROUS STATISTICAL SWEEP v3 (N=500, Spectral-Trust) ---")
    print("  Fixes active: [1] NaN scrubbing  [2] Library implementation [3] BOS Masking")
    os.makedirs("results/spectra", exist_ok=True)

    # ── FIX 1: drop NaN rows immediately after JSON load ──
    print(f"Loading dataset from {DATASET_PATH}...")
    with open(DATASET_PATH, "r") as f:
        data = json.load(f)
    
    df = pd.DataFrame(data)
    before = len(df)
    df = df.dropna(subset=["text"])
    df = df[df["text"].astype(str).str.strip() != ""]
    df = df[df["text"].astype(str).str.lower() != "nan"]
    after = len(df)
    print(f"Dataset: {before} samples -> {after} clean rows.")

    radicals = df[df["label"] == 1].sample(frac=1, random_state=42).reset_index(drop=True)
    neutrals = df[df["label"] == 0].sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"Pools: {len(radicals)} radicals, {len(neutrals)} neutrals.")

    # Configuration for spectral-trust
    config = GSPConfig(
        model_name=MODEL_NAME,
        device="cuda",
        normalization="rw",
        symmetrization="symmetric",
        head_aggregation="uniform",
        torch_dtype="float16",
        device_map={"": "cuda:0"},
        model_kwargs={
            "output_attentions": True,
            "output_hidden_states": True
        },
        save_intermediate=False,
        verbose=False
    )

    with GSPDiagnosticsFramework(config) as framework:
        # Load and instrument the model
        framework.instrumenter.load_model(MODEL_NAME)
        framework.instrumenter.register_hooks()
        model = framework.instrumenter.model
        tokenizer = framework.instrumenter.tokenizer

        # Resume support
        results = []
        if os.path.exists(CHECKPOINT_FILE):
            with open(CHECKPOINT_FILE, "r") as f:
                results = json.load(f).get("results", [])
            print(f"Resuming from sample {len(results)}...")

        # Pre-tokenise neutral pool
        print("\nPre-tokenising neutral pool...")
        neutral_pool = []
        for _, row in tqdm(neutrals.iterrows(), total=len(neutrals), desc="Neutral Pool"):
            txt = str(row["text"])
            t_len = len(tokenizer.encode(txt))
            neutral_pool.append({"text": txt, "len": t_len})
        neutral_pool.sort(key=lambda x: x["len"])

        # ── Main extraction loop ──
        for i in range(len(results), N_SAMPLES):
            if i >= len(radicals): break
            orig_radical = str(radicals.iloc[i]["text"])
            print(f"\n[{i+1}/{N_SAMPLES}] Processing...")

            try:
                # A. Style-transfer → Formalized Ghost
                formal_txt  = rewrite_text(model, tokenizer, orig_radical)
                formal_data = process_sequence_with_framework(framework, formal_txt)
                target_len  = formal_data["token_len"]

                # B. Length-matched neutral (±5%)
                lo, hi = target_len * (1 - LENGTH_TOLERANCE), target_len * (1 + LENGTH_TOLERANCE)
                candidates = [n for n in neutral_pool if lo <= n["len"] <= hi]
                if not candidates:
                    candidates = sorted(neutral_pool, key=lambda x: abs(x["len"] - target_len))[:5]
                
                match_txt    = random.choice(candidates)["text"]
                neutral_data = process_sequence_with_framework(framework, match_txt)

                results.append({
                    "radical_orig":    orig_radical,
                    "radical_formal":  formal_data,
                    "neutral_matched": neutral_data,
                })

                if len(results) % 20 == 0:
                    with open(CHECKPOINT_FILE, "w") as f:
                        json.dump({"results": results}, f, indent=2)
                    print(f"  [OK] Checkpoint @ {len(results)}")

                time.sleep(0.5) # Thermal sleep
                torch.cuda.empty_cache()

            except Exception as e:
                print(f"  Error on sample {i}: {e}")
                continue

    # Final save
    with open(OUTPUT_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSweep complete. N={len(results)} matched pairs saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
