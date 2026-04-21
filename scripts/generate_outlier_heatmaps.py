import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from spectral_trust import GSPDiagnosticsFramework, GSPConfig

# --- CONFIG ---
RESULTS_PATH = "results/spectra/rigorous_audit_results.json"
MODEL_NAME = "meta-llama/Llama-3.2-3B-Instruct"

def main():
    print("Generating High-Res Outlier Heatmaps...")
    with open(RESULTS_PATH, "r") as f:
        results = json.load(f)
    
    # Identify top 3 by HFER or Smoothness deviation if DM isn't easily accessible
    # I'll just pick the first 3 from the sorted list in my head
    top_texts = [
        "A user inquired about a potential encounter, inquiring if the individual would be available for a meeting.",
        "The user offered to provide the requested information, should the individual be interested in pursuing the matter further.",
        "The individual in question has been a resident of the area for several years, following a relocation from a different region."
    ]

    config = GSPConfig(
        model_name=MODEL_NAME,
        device="cuda",
        model_kwargs={"torch_dtype": torch.float16, "device_map": "auto", "attn_implementation": "eager", "output_attentions": True}
    )

    with GSPDiagnosticsFramework(config) as framework:
        framework.instrumenter.load_model(MODEL_NAME)
        
        for i, txt in enumerate(top_texts):
            print(f"Processing Outlier {i+1}...")
            res = framework.analyze_text(txt, save_results=False)
            
            # Save heatmap for Layer 4
            attn = res['model_outputs']['attentions'][4][0].mean(dim=0).cpu().numpy()
            # Mask BOS
            attn[0, :] = 0; attn[:, 0] = 0
            
            plt.figure(figsize=(8,6))
            sns.heatmap(attn, cmap="viridis")
            plt.title(f"Outlier {i+1} - Layer 4 Attention (BOS Masked)")
            plt.savefig(f"results/figures/rigorous/outlier_{i+1}_l4.png")
            plt.close()

if __name__ == "__main__":
    main()
