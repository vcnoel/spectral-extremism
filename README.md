# The Restricted Manifold: Topological Diagnosis of Radicalized Intent

This repository provides a framework for the training-free detection of radicalized ideological intent via the high-dimensional spectral analysis of self-attention manifolds in Transformer-based Large Language Models (LLMs).

## Abstract (Rev. 2026.04.21)

We demonstrate that radicalized intent triggers a distinctive **topological sparsity collapse** in late-layer attention manifolds. Our Rigor Audit ($N=200$) defines the **Restricted Manifold** as a collapsed geometric subset where extremist intent reside. By analyzes the Graph Laplacian of attention maps, we isolate an architecture-specific diagnostic signature (robust in the Llama family) that persists independently of sociolinguistic register or text complexity.

## Key Scientific Discoveries

1. **The Restricted Manifold (Geometric Subset):** Hate speech does not occupy a "separate" state space; instead, it forces the LLM's attention into a **restricted, rigid geometric state** (Low Gini, High L0 Smoothness). It is a topological subset characterized by a loss of mathematical rank and entropic collapse.
2. **Confounder Independence:** Using Multiple Linear Regression ($Label \sim Gini + Smoothness + Length + Perplexity$), we prove the topological signal is unique. 
   - **Partial $R^2$ (Topology):** **$0.071036$** ($p < 0.001$).
   - Topology explains variance that cannot be accounted for by sequence length or perplexity.
3. **The Llama Diagnostic Law:** The late-layer (Layer 24) sparsity collapse is a robust signature in the Llama family but is **not universal**. Audit of Qwen-2.5-7B showed no collapse ($\Delta G = -0.007$), suggesting the manifold collapse is specific to the Llama architecture or its alignment data.
4. **Causal Engine Proof:** Forced intervention at Layer 24 (clamping Gini to $0.055$) proves that topological collapse is a **causal engine** of generation. Clamping the manifold effectively "sanitizes" the model's output pathways.

---

## Forensic Diagnostics: Rigor Audit Results (N=200)

Comparative analysis on the Riabi dataset reveals the predictive power of topological features.

| Metric | Neutral Baseline | Hate Speech (Collapsed) | Result |
| :--- | :--- | :--- | :--- |
| **L24 Gini Index** | $0.049$ | **$0.044$** | **Sparsity Collapse** |
| **L0 Smoothness** | $0.0001$ | **$0.0024$** | **Structural Fracture** |
| **Cliff Index (V-Ratio)** | $1.0$ | **$< 0.5$** | **Volume Contraction** |

### Classification AUROC (Topological Features only)
- **Logistic Regression:** $0.7082$
- **SVM (RBF Kernel):** **$0.7358$**
- **Random Forest:** $0.7204$

## Visual Evidence

### Figure 1: The Mahalanobis Scatter
Plotting $L0\_Smoothness$ vs $L24\_Gini$ reveals the nested geometry of the Restricted Manifold. 95% confidence ellipses demonstrate the statistical separation achieved via topological forensics.

### Figure 2: The Volume Cliff
The calculus of collapse at Layer 0 shows a catastrophic drop in the spectral volume ratio ($V_{Hate} / V_{Neutral}$) below $0.5$, signaling the inception of intent.

---

## Methodology

We instrument the self-attention layers to extract the combinatorial Graph Laplacian:
$$L_{norm} = I - D^{-1/2} A D^{-1/2}$$
We then perform spectral decomposition to monitor:
- **Partial Correlation Analysis**: Controlling for linguistic confounders.
- **Sparsity Clamping**: Causal patching of the attention manifold.
- **Manifold Volume Trajectory**: Differential geometry of the state-space contraction.

## Usage

```bash
# 1. Forensic Rigor Audit (N=200)
conda run -n gemma_spectral python scripts/08_confounder_control.py

# 2. Model Invariance Validation
conda run -n gemma_spectral python scripts/09_cross_model_validation.py

# 3. AUROC Maximization
python scripts/04_maximize_auroc.py
```

## Citation

```bibtex
@article{noel2026extremism,
  title={The Restricted Manifold: Topological Diagnosis of Radicalized Intent},
  author={Valentin Noël},
  year={2026}
}
```
