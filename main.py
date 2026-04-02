import argparse
import json
import os
import subprocess
import random
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer

MODELS = [
    "meta-llama/Llama-3.2-1B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
    "Qwen/Qwen2.5-0.5B-Instruct",
    "microsoft/Phi-3.5-mini-instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
]

def cmd_build_dataset(args):
    """Download real corpora and assemble the clean register-controlled extremism dataset."""
    from datasets import load_dataset
    random.seed(42)
    
    print("Building Clean Register-Controlled Dataset...")
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
    
    categories = {
        "mhs_radical": [],
        "mhs_neutral": [],
        "stormfront_radical": [],
        "stormfront_neutral": [],
        "toxic_control": [],
        "wikipedia_formal": []
    }

    # 1. Stormfront (SetFit/hate_speech18) - 0: noHate, 1: hate
    try:
        print("Loading Stormfront (SetFit/hate_speech18)...")
        import requests
        sf_ds = load_dataset("json", data_files="https://huggingface.co/datasets/SetFit/hate_speech18/resolve/main/train.jsonl")["train"]
        sf_rad = [t["text"] for t in sf_ds if t["label"] == 1]
        sf_neu = [t["text"] for t in sf_ds if t["label"] == 0]
        random.shuffle(sf_rad)
        random.shuffle(sf_neu)
        for t in sf_rad:
            if len(tokenizer.encode(t)) > 15:
                categories["stormfront_radical"].append({"text": t, "label": 1, "category": "stormfront_radical"})
            if len(categories["stormfront_radical"]) >= 100: break
        for t in sf_neu:
            if len(tokenizer.encode(t)) > 15:
                categories["stormfront_neutral"].append({"text": t, "label": 0, "category": "stormfront_neutral"})
            if len(categories["stormfront_neutral"]) >= 100: break
    except Exception as e: print(f"Stormfront error: {e}")

    # 2. UC Berkeley MHS - using Parquet builder directly
    try:
        print("Loading MHS (ucberkeley-dlab/measuring-hate-speech)...")
        mhs_ds = load_dataset("parquet", data_files="https://huggingface.co/datasets/ucberkeley-dlab/measuring-hate-speech/resolve/main/measuring-hate-speech.parquet")["train"]
        mhs_rad = [t["text"] for t in mhs_ds if t["hate_speech_score"] > 1.0]
        mhs_neu = [t["text"] for t in mhs_ds if t["hate_speech_score"] < -1.0]
        random.shuffle(mhs_rad)
        random.shuffle(mhs_neu)
        for t in mhs_rad:
            if 15 < len(tokenizer.encode(t)) < 150:
                categories["mhs_radical"].append({"text": t, "label": 1, "category": "mhs_radical"})
            if len(categories["mhs_radical"]) >= 100: break
        for t in mhs_neu:
            if 15 < len(tokenizer.encode(t)) < 150:
                categories["mhs_neutral"].append({"text": t, "label": 0, "category": "mhs_neutral"})
            if len(categories["mhs_neutral"]) >= 100: break
    except Exception as e: print(f"MHS error: {e}")

    # 3. Jigsaw Controls (SetFit/toxic_conversations)
    try:
        print("Loading Jigsaw Controls (SetFit/toxic_conversations)...")
        # label 1: toxic, label 0: clean
        j_ds = load_dataset("json", data_files="https://huggingface.co/datasets/SetFit/toxic_conversations/resolve/main/train.jsonl")["train"]
        toxic_samples = [t["text"] for t in j_ds if t["label"] == 1]
        clean_samples = [t["text"] for t in j_ds if t["label"] == 0]
        random.shuffle(toxic_samples)
        random.shuffle(clean_samples)
        
        # Toxic Control
        for t in toxic_samples:
            if len(tokenizer.encode(t)) > 20:
                categories["toxic_control"].append({"text": t, "label": 0, "category": "toxic_control"})
            if len(categories["toxic_control"]) >= 100: break
            
        # Wikipedia Formal (Clean samples with length and low-punctuation heuristics)
        for t in clean_samples:
            words = t.split()
            if len(words) > 40 and not any(char in t for char in "!?"):
                categories["wikipedia_formal"].append({"text": t, "label": 0, "category": "wikipedia_formal"})
            if len(categories["wikipedia_formal"]) >= 100: break
            
    except Exception as e: print(f"Controls error: {e}")

    # Assemble
    final_dataset = []
    print("\nDataset Summary:")
    for cat, samples in categories.items():
        count = len(samples)
        final_dataset.extend(samples)
        print(f" - {cat}: {count}")

    output_path = Path("data/extremism_dataset.json")
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(final_dataset, f, indent=2)
    print(f"\nFinal Corpus Size: N={len(final_dataset)}")

def cmd_extract(args):
    """Run spectral extraction and statistical analysis (including Nested CV)."""
    cmd = ["python", "scripts/run_extremism.py", "--model", args.model]
    if args.results_file: cmd.extend(["--results-file", args.results_file])
    if args.load_in_4bit: cmd.append("--load-in-4bit")
    if args.load_in_8bit: cmd.append("--load-in-8bit")
    subprocess.run(cmd)

def cmd_categories(args):
    cmd = ["python", "scripts/analyze_categories.py", "--results", args.results]
    subprocess.run(cmd)

def main():
    parser = argparse.ArgumentParser(description="Spectral Geometry of Extremism")
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("build-dataset")
    
    p_ext = sub.add_parser("extract")
    p_ext.add_argument("--model", required=True, help="HuggingFace model ID")
    p_ext.add_argument("--results-file", help="Custom results path")
    p_ext.add_argument("--load-in-4bit", action="store_true")
    p_ext.add_argument("--load-in-8bit", action="store_true")

    p_cat = sub.add_parser("categories")
    p_cat.add_argument("--results", required=True, help="Path to results JSON")

    args = parser.parse_args()
    if args.command == "build-dataset":
        cmd_build_dataset(args)
    elif args.command == "extract":
        cmd_extract(args)
    elif args.command == "categories":
        cmd_categories(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
