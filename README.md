# Spectral Geometry of Extremism

Training-free detection of radicalized text via spectral analysis of transformer attention graphs.

## Key Findings

This benchmark demonstrates that radicalized text exhibits a distinctive **spectral-topological collapse** within Large Language Models (LLMs). By analyzing the attention manifolds of 5 diverse architectures (from 0.5B to 8B parameters), we found:
- **Trajectory Superiority**: The dynamic evolution of the spectral signal is significantly more discriminative than any single layer. Our full-trajectory modeling achieves **0.75–0.82 AUROC**, consistently outperforming single-layer discrimination.
- **Register Control**: We discovered a massive "informality confound" where models initially appeared to detect radicalization but were actually detecting the formal register of Wikipedia. Register-controlled comparisons reveal a genuine but subtler extremism signal (Cohen's $d \approx 0.3-0.8$).
- **Scale Invariance**: The spectral signature of radicalization (falling High-Frequency Energy Ratio) is preserved across weights-only and instruction-tuned variants, suggesting it is a fundamental property of transformer-based knowledge representations.

## Method

We use the `spectral-trust` library to instrument the model's self-attention layers:
1. **Symmetrized Attention**: For each text input, we extract the $N \times N$ attention matrix ($A$) at each layer and symmetrize it: $W = 0.5(A + A^T)$. Heads are aggregated via mass-weighting.
2. **Combinatorial Laplacian**: We construct the combinatorial Graph Laplacian $L = D - W$.
3. **Spectral Diagnostics**:
   - **Fiedler Value**: The second smallest eigenvalue (algebraic connectivity).
   - **HFER (High-Frequency Energy Ratio)**: The ratio of signal energy (hidden states projected onto the eigenbasis) in the upper half of the eigenspectrum vs. the lower half. 
   - **Smoothness Index**: The quadratic form $x^T L x$ where $x$ are the hidden states.
4. **Trajectory Mapping**: We analyze the full layer-wise profile of these metrics, processing the 112-dimensional vector ($4 \text{ metrics} \times \text{layers}$) to capture terminal topological contractions.

## Results (MHS Gold Standard Benchmark)

We achieved a significant performance breakthrough by moving from "summary statistics" to **Full-Trajectory Modeling** using Logistic Regression on the complete 112-dimensional spectral signature.

### 5-Model AUROC Comparison: Full Trajectory Analysis

| Model | Hand-crafted AUROC | **Full Traj LogReg AUROC** | Full Traj SVM AUROC |
| :--- | :--- | :--- | :--- |
| **Llama-3.2-3B** | 0.769 | **0.825** | 0.828 |
| **Llama-3.1-8B** | 0.731 | **0.801** | 0.812 |
| **Mistral-7B** | 0.710 | **0.785** | 0.792 |
| **Qwen-2.5-0.5B** | 0.703 | **0.765** | 0.771 |
| **Llama-3.2-1B** | 0.695 | **0.758** | 0.762 |

### Mechanistic Interpretation (Llama-3.2-3B)

By analyzing the linear coefficients of the LogReg model, we've mapped exactly which "spectral hotspots" define the extremism signature:

| Feature Rank | (Metric, Layer) | Weight | Interpretation |
| :--- | :--- | :--- | :--- |
| **Top Positive** | Smoothness @ L26 | +1.39 | **Delayed Collapse**: Sudden terminal topological contraction. |
| **Top Positive** | Entropy @ L0 | +1.04 | **Embedding Variance**: Initial dispersed state distribution. |
| **Top Negative** | Smoothness @ L14 | -1.48 | **Mid-Layer Stability**: Strong indicator of neutral/formal registers. |

## The Register Confound

Our most important methodological finding: initial experiments showed $d > 1.0$, but this was largely a register artifact. Models distinguish formal (Wikipedia) from informal (social media) text with $d \approx 1.0–1.5$, regardless of ideological content.

After register-controlled comparisons (radical vs. neutral from the SAME source):
- **Genuine extremism signal**: $d \approx 0.3–0.8$ (MHS within-source)
- **Linguistic deviation noise**: $d \approx 0.7–1.3$ (repetitive trolling vs. normal text)
- **Register artifact**: $d \approx 1.0–1.5$ (formal vs. informal)

This decomposition is critical for any future work on spectral content detection.

## Dataset

We use a register-controlled composite corpus (N=600):
- **MHS Within-Source**: Radical/Neutral pairs from the *Berkeley Measuring Hate Speech* survey. Highly controlled register.
- **Stormfront Within-Source**: Ideologically pure samples from Stormfront forum vs. neutral forum text.
- **Toxic/Formal Controls**: Wikipedia (Formal) and Toxic comments (Trolling) to isolate the radicalization signal from register and toxicity effects.

## Trajectory Visualizations

Below are the spectral profiles for the top-performing models:

#### Llama-3.2-3B-Instruct
![Llama-3.2-3B Trajectories](results/figures/trajectories_Llama-3.2-3B-Instruct.png)

#### Llama-3.1-8B-Instruct
![Llama-3.1-8B Trajectories](results/figures/trajectories_Llama-3.1-8B-Instruct.png)

## Usage

### 1. Build the Dataset
```bash
python main.py build-dataset
```

### 2. Run Extraction (Single Model)
```bash
python main.py extract --model meta-llama/Llama-3.2-3B-Instruct
```

### 3. Analyze Features
```bash
python main.py features --results results/spectra/extremism_results_Llama-3.2-3B-Instruct.json
```

## Limitations

- The genuine within-source extremism signal is modest ($d \approx 0.3–0.8$). This is sufficient for research analysis but not for deployment as a standalone detector.
- Dataset is English-only. Extension to French and Arabic is planned.
- Full trajectory logistic regression uses 112 features on 200 samples—risk of overfitting despite L2 regularization. The 5-fold CV mitigates but does not eliminate this.
- Phi-3.5 and Qwen-MoE were not tested due to hardware constraints.

## Citation

```bibtex
@article{noel2026extremism,
  title={Spectral Geometry of Extremism: Trajectory-Level Detection in Transformers},
  author={Valentin Noël},
  journal={Internal Report},
  year={2026}
}

@inproceedings{noel2026geometry,
  title={The Geometry of Reason},
  author={Valentin Noël},
  year={2026},
  note={Under Review}
}

@article{noel2026archaeology,
  title={Spectral Archaeology: The Causal Topology of Model Evolution},
  author={Valentin Noël},
  year={2026},
  note={Under Review}
}
```

## Acknowledgments
- `spectral-trust`: Core graph-spectral instrumentation.
- UC Berkeley D-Lab: Measuring Hate Speech dataset.
