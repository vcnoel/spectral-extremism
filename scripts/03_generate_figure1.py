import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
import os

def confidence_ellipse(x, y, ax, n_std=3.0, facecolor='none', **kwargs):
    """
    Create a plot of the covariance confidence ellipse of *x* and *y*.
    """
    if x.size != y.size:
        raise ValueError("x and y must be the same size")

    cov = np.cov(x, y)
    pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
    # Using a special case to obtain the eigenvalues of this
    # 2D dataset.
    ell_radius_x = np.sqrt(1 + pearson)
    ell_radius_y = np.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)

    # Calculating the standard deviation of x from
    # the square-root of the variance and multiplying
    # with the given number of standard deviations.
    scale_x = np.sqrt(cov[0, 0]) * n_std
    mean_x = np.mean(x)

    # calculating the standard deviation of y ...
    scale_y = np.sqrt(cov[1, 1]) * n_std
    mean_y = np.mean(y)

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)

def main():
    # Load data
    df = pd.read_csv('data/riabi_features_N200.csv')
    
    # Filter groups
    neutral = df[df['label'] == 'Neutral']
    radical = df[df['label'] == 'Hate Speech']
    
    # Calculate means for reporting
    mu_n_gini = neutral['L24_gini'].mean()
    mu_r_gini = radical['L24_gini'].mean()
    
    print(f"Mean L24_gini (Neutral): {mu_n_gini:.6f}")
    print(f"Mean L24_gini (Hate Speech): {mu_r_gini:.6f}")
    
    # Plotting
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except:
        plt.style.use('whitegrid')
        
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Scatter plots
    ax.scatter(neutral['L0_smoothness'], neutral['L24_gini'], 
               color='blue', marker='o', alpha=0.6, label='Neutral', s=50)
    ax.scatter(radical['L0_smoothness'], radical['L24_gini'], 
               color='red', marker='x', alpha=0.8, label='Hate Speech', s=60)
    
    # Ellipses (95% confidence -> ~2.45 std dev for 2D normal)
    # The user asked for 95%. For a 2D normal distribution, 95% is about 2.447 sigmas.
    confidence_ellipse(neutral['L0_smoothness'], neutral['L24_gini'], ax, 
                       n_std=2.447, edgecolor='blue', linestyle='--', linewidth=2, label='95% CI (Neutral)')
    confidence_ellipse(radical['L0_smoothness'], radical['L24_gini'], ax, 
                       n_std=2.447, edgecolor='red', linestyle='--', linewidth=2, label='95% CI (Hate Speech)')
    
    # Formatting
    ax.set_title('Topological Separation of Radicalized Intent (Riabi Dataset)', fontsize=15, pad=20)
    ax.set_xlabel('L0 Smoothness Index ($\phi_0$)', fontsize=12)
    ax.set_ylabel('L24 Gini Sparsity ($G_{24}$)', fontsize=12)
    ax.legend(frameon=True, facecolor='white', framealpha=1)
    
    # Output
    output_dir = 'results/figures/forensic'
    os.makedirs(output_dir, exist_ok=True)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/figure1_mahalanobis.png', dpi=300)
    plt.savefig(f'{output_dir}/figure1_mahalanobis.pdf')
    
    print(f"Successfully generated figures in {output_dir}")

if __name__ == "__main__":
    main()
