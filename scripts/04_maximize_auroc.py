import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, IsolationForest
from sklearn.metrics import roc_auc_score, make_scorer
import warnings
warnings.filterwarnings('ignore')

def main():
    # Load data
    df = pd.read_csv('data/riabi_features_N200.csv')
    df = df.dropna(axis=1, how='all')
    
    label_col = 'label'
    y = (df[label_col] == 'Hate Speech').astype(int)
    
    # Get top 10 discriminative features (based on previous run)
    # We'll calculate them again to be sure, or just use the same logic
    features = [c for c in df.columns if any(p in c for p in ['L', 'max_velocity'])]
    
    neutral = df[df[label_col] == 'Neutral']
    radical = df[df[label_col] == 'Hate Speech']
    
    univariate_results = []
    for f in features:
        r_vals = radical[f].values
        n_vals = neutral[f].values
        n1, n2 = len(r_vals), len(n_vals)
        var1, var2 = np.var(r_vals, ddof=1), np.var(n_vals, ddof=1)
        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
        d = (np.mean(r_vals) - np.mean(n_vals)) / pooled_std if pooled_std != 0 else 0
        univariate_results.append({'Feature': f, 'abs_d': abs(d)})
    
    top_10_features = pd.DataFrame(univariate_results).sort_values('abs_d', ascending=False).head(10)['Feature'].tolist()
    X = df[top_10_features].values
    
    # Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Models
    models = {
        "SVM (RBF)": SVC(kernel='rbf', probability=True),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42)
    }
    
    # XGBoost check
    try:
        from xgboost import XGBClassifier
        models["XGBoost"] = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    except ImportError:
        pass

    results = {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    for name, model in models.items():
        scores = cross_val_score(model, X_scaled, y, cv=cv, scoring='roc_auc')
        results[name] = np.mean(scores)
        
    # Isolation Forest (Anomaly Detection)
    # Treat Neutral (0) as normal, Hate Speech (1) as anomaly
    # Scikit-learn IsolationForest returns -1 for anomalies, 1 for normal.
    # To get a score that acts like a probability for 'Hate Speech', we use decision_function.
    # decision_function: the lower, the more abnormal. So we take -decision_function.
    iso = IsolationForest(contamination=0.5, random_state=42)
    # We train on full set (since it's unsupervised mainly) or do CV.
    # For fair AUROC comparison, we can do it on the whole set or CV.
    # Let's do it on the whole set for simplicity or CV.
    
    def iso_scorer(estimator, X, y):
        # Isolation Forest's score_samples or -decision_function
        return roc_auc_score(y, -estimator.decision_function(X))
    
    iso_scores = cross_val_score(iso, X_scaled, y, cv=cv, scoring=iso_scorer)
    results["Isolation Forest"] = np.mean(iso_scores)
    
    # Output table
    print(f"\n{'Algorithm':<25} | {'Mean AUROC':<10}")
    print("-" * 38)
    print(f"{'Logistic Regression (Base)':<25} | {0.7080:<10.4f}")
    for name, score in results.items():
        print(f"{name:<25} | {score:<10.4f}")

if __name__ == "__main__":
    main()
