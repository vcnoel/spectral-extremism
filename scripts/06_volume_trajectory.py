import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def calculate_manifold_volume(df, layer_idx, epsilon=1e-6):
    metrics = ['energy', 'smoothness', 'entropy', 'hfer', 'fiedler']
    cols = [f'L{layer_idx}_{m}' for m in metrics]
    
    # Check if columns exist
    if not all(c in df.columns for c in cols):
        return None
        
    data = df[cols].dropna()
    if len(data) < len(metrics):
        return None
        
    # Covariance matrix
    cov = np.cov(data.values.T)
    
    # Regularization (Tikhonov)
    cov_reg = cov + epsilon * np.eye(cov.shape[0])
    
    # Determinant (Volume)
    # Using slogdet for numerical stability (returns sign and log of determinant)
    sign, logdet = np.linalg.slogdet(cov_reg)
    return np.exp(logdet)

def main():
    # Load data
    csv_path = 'data/riabi_features_N200.csv'
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return
        
    df = pd.read_csv(csv_path)
    
    neutral_df = df[df['label'] == 'Neutral']
    hate_df = df[df['label'] == 'Hate Speech']
    
    num_layers = 28
    results = []
    
    for i in range(num_layers):
        v_neutral = calculate_manifold_volume(neutral_df, i)
        v_hate = calculate_manifold_volume(hate_df, i)
        
        if v_neutral is not None and v_hate is not None:
            ratio = v_hate / v_neutral
            results.append({'Layer': i, 'Ratio': ratio, 'V_Neutral': v_neutral, 'V_Hate': v_hate})
            
    res_df = pd.DataFrame(results)
    
    # Find the "Cliff"
    cliff_index = res_df[res_df['Ratio'] < 0.5]['Layer']
    if not cliff_index.empty:
        cliff_layer = cliff_index.iloc[0]
        print(f"CLIFF_INDEX: {cliff_layer}")
    else:
        print("CLIFF_INDEX: Not reached (< 0.5)")
        
    # Plotting
    plt.figure(figsize=(10, 6))
    sns.set_style('whitegrid')
    plt.plot(res_df['Layer'], res_df['Ratio'], marker='o', linestyle='-', color='crimson')
    plt.axhline(y=0.5, color='gray', linestyle='--', label='Cliff Threshold (0.5)')
    plt.title('Manifold Volume Ratio (Hate/Neutral) Across Layers', fontsize=14)
    plt.xlabel('Layer Index', fontsize=12)
    plt.ylabel('Volume Ratio ($V_{Hate} / V_{Neutral}$)', fontsize=12)
    plt.yscale('log') # Use log scale if ratios vary wildly
    plt.legend()
    
    output_dir = 'results/figures/forensic'
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f'{output_dir}/manifold_volume_trajectory.png', dpi=300)
    plt.savefig(f'{output_dir}/manifold_volume_trajectory.pdf')
    print(f"Results saved to {output_dir}")

if __name__ == "__main__":
    main()
