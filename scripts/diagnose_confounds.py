import json
import numpy as np
import random
from transformers import AutoTokenizer

def diagnose():
    # Load data
    with open('data/extremism_dataset.json', 'r') as f:
        dataset = json.load(f)
    
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
    
    # Categorize raw data
    categories = {}
    for item in dataset:
        cat = item['category']
        if cat not in categories:
            categories[cat] = []
        categories[cat].append(item)
        
    print("="*80)
    print("CATEGORY TOKEN LENGTH STATISTICS")
    print("="*80)
    print(f"{'Category':<25} | {'Mean':<6} | {'Median':<6} | {'Std':<6} | {'Min':<4} | {'Max':<4}")
    print("-" * 80)
    
    for cat, items in sorted(categories.items()):
        lengths = [len(tokenizer.encode(item['text'])) for item in items]
        mean_len = np.mean(lengths)
        median_len = np.median(lengths)
        std_len = np.std(lengths)
        min_len = min(lengths)
        max_len = max(lengths)
        print(f"{cat:<25} | {mean_len:6.1f} | {median_len:6.1f} | {std_len:6.1f} | {min_len:4} | {max_len:4}")
    
    print("\n" + "="*80)
    print("RANDOM TEXT SAMPLES: JIGSAW_RADICAL")
    print("="*80)
    j_rad = [item['text'] for item in categories.get('jigsaw_radical', [])]
    for i, text in enumerate(random.sample(j_rad, min(len(j_rad), 5))):
        print(f"{i+1}. {text[:500]}...")
        print("-" * 40)
        
    print("\n" + "="*80)
    print("RANDOM TEXT SAMPLES: JIGSAW_NEUTRAL_INFORMAL")
    print("="*80)
    j_neu = [item['text'] for item in categories.get('jigsaw_neutral_informal', [])]
    for i, text in enumerate(random.sample(j_neu, min(len(j_neu), 5))):
        print(f"{i+1}. {text[:500]}...")
        print("-" * 40)

if __name__ == "__main__":
    diagnose()
