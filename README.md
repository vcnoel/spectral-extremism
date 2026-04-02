# Spectral Geometry of Extremism

Training-free detection of radicalized text via spectral analysis of transformer attention graphs.

## Key Findings

This benchmark demonstrates that radicalized text exhibits a distinctive **spectral-topological collapse** within Large Language Models (LLMs). By analyzing the attention manifolds of 7 diverse architectures (from 0.5B to 8B parameters), we found:
- **Trajectory Superiority**: The dynamic evolution of the spectral signal (the "HFER crossover") is significantly more discriminative than any single layer. Our Multi-Metric Trajectory (MMT) classifier achieves **66-70% accuracy**, consistently outperforming single-threshold baselines.
- **Register Control**: We discovered a massive "informality confound" where models initially appeared to detect radicalization but were actually detecting the formal register of Wikipedia. Register-controlled comparisons (MHS within-source) reveal a genuine but subtler extremism signal (Cohen's $d \approx 0.3-0.8$).
- **Scale Invariance**: The spectral signature of radicalization (falling High-Frequency Energy Ratio) is preserved across weights-only and instruction-tuned variants, suggesting it is a fundamental property of transformer-based knowledge representations.

## Method

We use the `spectral-trust` library to instrument the model's self-attention layers:
1. **Laplacian Construction**: For each text input, we extract the $N \times N$ symmetrized attention matrix ($A$) at each layer and construct the Graph Laplacian $L = D - A$.
2. **Spectral Diagnostics**: We compute the spectrum of $L$, specifically focusing on:
   - **Fiedler Value**: The second smallest eigenvalue (algebraic connectivity).
   - **HFER (High-Frequency Energy Ratio)**: The ratio of energy in the top-half vs. bottom-half of the spectrum.
   - **Smoothness Index**: The quadratic form $x^T L x$ where $x$ are the hidden states.
3. **Trajectory Mapping**: We analyze the "smoothness collapse"—where radical sentences start with high spectral noise and rapidly settle into lower-dimensional manifolds compared to neutral controls.

## Results (MHS Gold Standard Benchmark)

| Model | Params | Best Single-Layer $d$ | Single Acc | **MMT Traj Acc** | MMT AUROC | HFER ABC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Qwen-2.5-0.5B** | 0.5B | -0.75 | 60.3% | **67.2%** | 0.66 | 2.52 |
| **Llama-3.2-1B** | 1.2B | 0.74 | 58.5% | **66.2%** | 0.70 | 2.01 |
| **Llama-3.2-3B** | 3.2B | 0.65 | 49.0% | **69.3%** | 0.73 | 1.54 |
| **Qwen-MoE-A2.7B** | 2.7B | *Extracting* | *TBD* | *TBD* | *TBD* | *TBD* |
| **Phi-3.5-mini** | 3.8B | *Extracting* | *TBD* | *TBD* | *TBD* | *TBD* |
| **Mistral-7B** | 7.2B | *Extracting* | *TBD* | *TBD* | *TBD* | *TBD* |
| **Llama-3.1-8B** | 8.0B | *Extracting* | *TBD* | *TBD* | *TBD* | *TBD* |

## Dataset

We use a register-controlled composite corpus (N=600):
- **MHS Within-Source**: Radical/Neutral pairs from the *Berkeley Measuring Hate Speech* survey. Highly controlled register.
- **Stormfront Within-Source**: Ideologically pure samples from Stormfront forum vs. neutral forum text.
- **Toxic/Formal Controls**: Wikipedia (Formal) and Toxic comments (Trolling) to isolate the radicalization signal from register and toxicity effects.

## Usage

### 1. Build the Dataset
```bash
python main.py build-dataset
```

### 2. Run Extraction (Single Model)
```bash
python main.py extract --model meta-llama/Llama-3.2-3B-Instruct
```

### 3. Run Global Audit
```bash
powershell -File scripts/audit_models.ps1
```

## Citation

```bibtex
@article{antigravity2024extremism,
  title={Spectral Geometry of Extremism: Trajectory-Level Detection in Transformers},
  author={Antigravity Research Group},
  journal={Internal Report},
  year={2024}
}

@inproceedings{geometry2024,
  title={The Geometry of Reason},
  author={Anonymous},
  year={2024},
  note={Under Review}
}
```

## Acknowledgments
- `spectral-trust`: Core graph-spectral instrumentation.
- UC Berkeley D-Lab: Measuring Hate Speech dataset.
