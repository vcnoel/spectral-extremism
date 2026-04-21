import json
import random
import numpy as np

def curate():
    with open('data/extremism_dataset.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Task 2: Human Manifesto (Stormfront/MHS)
    rads = [d for d in data if d['label'] == 1 and d['category'].startswith(('stormfront', 'mhs'))]
    neutrals = [d for d in data if d['label'] == 0]
    
    # Sort rads by length to find "formal/long" ones
    rads.sort(key=lambda x: len(x['text'].split()), reverse=True)
    candidates = rads[:150] # Buffer
    
    curated_rads = []
    curated_neuts = []
    
    # Heuristic matching: for each top rad, find a neutral with similar word count (+/- 5%)
    for r in candidates:
        rlen = len(r['text'].split())
        # Find closest neutral
        matches = [n for n in neutrals if abs(len(n['text'].split()) - rlen) <= max(2, int(0.1*rlen))]
        if matches:
            curated_rads.append(r)
            n_match = random.choice(matches)
            curated_neuts.append(n_match)
            neutrals.remove(n_match) # Avoid reuse
        
        if len(curated_rads) >= 100:
            break
            
    print(f"Curated {len(curated_rads)} matched pairs.")
    avg_rad_len = np.mean([len(r['text'].split()) for r in curated_rads])
    avg_neu_len = np.mean([len(n['text'].split()) for n in curated_neuts])
    print(f"Avg Length - Radical: {avg_rad_len:.2f}, Neutral: {avg_neu_len:.2f}")
    
    # Save curation
    output = {
        "human_radical": curated_rads,
        "human_neutral": curated_neuts,
        "n_1000_pool": [d for d in data if d['label'] == 1][:1000],
        "en_neutral_centroid": [d for d in data if d['category'] == 'riabi_immigrants_en_neutral'],
        "es_radical": [d for d in data if d['category'] == 'riabi_immigrants_es_radical'],
        "it_radical": [d for d in data if d['category'] == 'riabi_immigrants_it_radical']
    }
    
    with open('data/curated_rigor_sweep.json', 'w') as f:
        json.dump(output, f, indent=2)

if __name__ == "__main__":
    curate()
