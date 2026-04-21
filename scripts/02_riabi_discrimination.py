import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

def calculate_cohens_d(group1, group2):
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std == 0: return 0.0
    return (np.mean(group1) - np.mean(group2)) / pooled_std

def run_analysis():
    df = pd.read_csv('data/riabi_features_N200.csv')
    df = df.dropna(axis=1, how='all') # Drop layers that might be empty if model has < 28 (Llama 3.2 3B has 28)
    
    label_col = 'label'
    classes = df[label_col].unique()
    neutral = df[df[label_col] == 'Neutral']
    radical = df[df[label_col] == 'Hate Speech']
    
    # Task 1: Univariate Top 10
    features = [c for c in df.columns if any(p in c for p in ['L', 'max_velocity'])]
    univariate_results = []
    
    for f in features:
        r_vals = radical[f].values
        n_vals = neutral[f].values
        
        # d
        d = calculate_cohens_d(r_vals, n_vals)
        # Wilcoxon
        try:
            stat, p = stats.ranksums(r_vals, n_vals)
        except:
            p = 1.0
            
        univariate_results.append({
            'Feature': f,
            'Cohen d': d,
            'P-value': p
        })
        
    uni_df = pd.DataFrame(univariate_results)
    uni_df['abs_d'] = uni_df['Cohen d'].abs()
    top_10 = uni_df.sort_values('abs_d', ascending=False).head(10)
    
    print("\n=== TOP 10 DISCRIMINATIVE FEATURES ===")
    print(top_10[['Feature', 'Cohen d', 'P-value']].to_string(index=False))
    
    # Task 2: Mahalanobis Distance (D) using Top 5
    top_5_features = top_10['Feature'].head(5).tolist()
    X_rad = radical[top_5_features].values
    X_neut = neutral[top_5_features].values
    
    mu_rad = np.mean(X_rad, axis=0)
    mu_neut = np.mean(X_neut, axis=0)
    S_rad = np.cov(X_rad, rowvar=False)
    S_neut = np.cov(X_neut, rowvar=False)
    
    n_r, n_n = len(X_rad), len(X_neut)
    S_pooled = ((n_r - 1) * S_rad + (n_n - 1) * S_neut) / (n_r + n_n - 2)
    # Add small epsilon to diagonal for stability
    S_pooled += np.eye(S_pooled.shape[0]) * 1e-6
    S_inv = np.linalg.inv(S_pooled)
    
    dist_m = np.sqrt((mu_rad - mu_neut).T @ S_inv @ (mu_rad - mu_neut))
    print(f"\nMahalanobis Distance (D, Top 5): {dist_m:.4f}")
    
    # Task 3: Upper Bound AUROC
    X = df[features].values
    y = (df[label_col] == 'Hate Speech').astype(int)
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Use Logistic Regression with 5-fold CV to get a robust AUROC
    clf = LogisticRegression(max_iter=1000)
    auroc_scores = cross_val_score(clf, X_scaled, y, cv=5, scoring='roc_auc')
    
    print(f"Zero-Shot AUROC (Upper Bound): {np.mean(auroc_scores):.4f} (+/- {np.std(auroc_scores):.4f})")

if __name__ == "__main__":
    run_analysis()
