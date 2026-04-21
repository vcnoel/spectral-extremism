import torch
import numpy as np
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

def inspect_attention():
    model_name = "meta-llama/Llama-3.2-3B-Instruct"
    print(f"Loading {model_name} for attention inspection...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        device_map="auto", 
        torch_dtype=torch.float16,
        attn_implementation="eager"
    )

    texts = [
        "The jurisdictional implications of the non-proliferation treaty regarding extraterritorial extraction of silicon-based lifeforms.",
        "Quantum chromodynamics suggests that the asymptotic freedom of gluons is a consequence of the non-abelian nature of the gauge group."
    ]

    for text in texts:
        print(f"\nInspecting: {text[:50]}...")
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model(**inputs, output_attentions=True)
        
        # Check layer 4 (inception layer)
        L = 4
        A = outputs.attentions[L][0].to(torch.float32).cpu().numpy().mean(axis=0)
        
        print(f"Matrix A (Layer {L}) shape: {A.shape}")
        print(f"Min: {np.min(A)}, Max: {np.max(A)}, Mean: {np.mean(A)}")
        print(f"Any NaN: {np.any(np.isnan(A))}, Any Inf: {np.any(np.isinf(A))}")
        print(f"Rank: {np.linalg.matrix_rank(A)}")
        
        # Check Laplacian properties
        D = np.diag(np.sum(A, axis=1))
        L_mat = D - A
        print(f"Laplacian sum row: {np.sum(L_mat, axis=1)}") # should be close to 0

if __name__ == "__main__":
    inspect_attention()
