import json
import numpy as np
from scipy.stats import pointbiserialr, pearsonr

def partial_corr(x, y, z):
    """Calculates partial correlation between x and y, controlling for z."""
    r_xy = pearsonr(x, y)[0]
    r_xz = pearsonr(x, z)[0]
    r_yz = pearsonr(y, z)[0]
    numerator = r_xy - (r_xz * r_yz)
    denominator = np.sqrt((1 - r_xz**2) * (1 - r_yz**2))
    return numerator / denominator

def run_correlation_study():
    metadata_path = 'data/results/rebuttal/llama8b_full_extraction.json'
    with open(metadata_path, 'r') as f:
        data = json.load(f)
        
    # Filtering samples with spectral data
    samples = [d for d in data if 'spectral' in d]
    
    valid = np.array([1 if d.get('is_valid') else 0 for d in samples])
    lengths = np.array([float(d.get('n_tokens', d.get('proof_token_count', 0))) for d in samples])
    
    print(f"--- RL REWARD CORRELATION ANALYSIS (N={len(samples)}) ---")
    
    for l in [0, 8, 16, 21, 24, 30]:
        hfers = np.array([d['spectral'][f'layer_{l}']['hfer'] for d in samples])
        
        # Point Biserial Correlation (HFER vs Valid)
        p_bis, _ = pointbiserialr(hfers, valid)
        
        # Partial Correlation (HFER vs Valid | Length)
        p_part = partial_corr(hfers, valid, lengths)
        
        # HFER vs Length correlation
        h_len_corr, _ = pearsonr(hfers, lengths)
        
        print(f"Layer {l}: HFER-Valid={p_bis:+.3f} | Partial(HFER-Valid|Length)={p_part:+.3f} | HFER-Length={h_len_corr:+.3f}")

    # Baseline: Mean Log-Prob
    lps = np.array([d['token_baselines']['mean_logprob'] for d in samples])
    lp_valid, _ = pointbiserialr(lps, valid)
    lp_len, _ = pearsonr(lps, lengths)
    lp_part = partial_corr(lps, valid, lengths)
    
    print("\n--- BASELINES ---")
    print(f"Mean Log-Prob: LP-Valid={lp_valid:+.3f} | Partial(LP-Valid|Length)={lp_part:+.3f} | LP-Length={lp_len:+.3f}")

if __name__ == "__main__":
    run_correlation_study()
