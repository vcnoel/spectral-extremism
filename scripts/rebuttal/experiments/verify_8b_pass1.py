import os
import json
import subprocess
import time
from tqdm import tqdm

def verify_lean_file(file_path):
    """Compiles a Lean 4 file and returns its validity."""
    try:
        # We assume 'lean' is in the path and works for Lean 4
        result = subprocess.run(
            ['lean', file_path],
            capture_output=True,
            text=True,
            timeout=10 # 10s timeout per proof should be plenty for MiniF2F
        )
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        return False
    except Exception:
        return False

def run_pass1_validation():
    metadata_path = 'data/results/rebuttal/llama8b_full_extraction.json'
    proofs_dir = 'data/proofs_minif2f'
    
    with open(metadata_path, 'r') as f:
        data = json.load(f)
    
    results = []
    print(f"Validating {len(data)} proofs via Lean 4 ground-truth...")
    
    valid_count = 0
    total_count = 0
    
    for item in tqdm(data):
        file_basename = item['file']
        # Locate the file (searching in valid_ground_truth and invalid)
        target_path = None
        for root, dirs, files in os.walk(proofs_dir):
            if file_basename in files:
                target_path = os.path.join(root, file_basename)
                break
        
        if not target_path:
            continue
            
        is_actually_valid = verify_lean_file(target_path)
        item['is_actually_valid'] = is_actually_valid
        
        if is_actually_valid:
            valid_count += 1
        total_count += 1
        results.append(item)
    
    # Final output
    output_path = 'data/results/rebuttal/verify_8b_pass1.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
        
    print("\n" + "="*50)
    print("  LEAN 4 GROUND-TRUTH PASS@1 SUMMARY")
    print("="*50)
    print(f"  Total Proofs Checked: {total_count}")
    print(f"  Valid (Compiles):     {valid_count}")
    print(f"  Ground-Truth Accuracy: {(valid_count/total_count):.2%}")
    print("="*50)

if __name__ == "__main__":
    run_pass1_validation()
