# OTDA-LSMM

[![arXiv](https://img.shields.io/badge/arXiv-2606.26482-b31b1b.svg)](https://arxiv.org/abs/2606.26482)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository provides the official implementation of the algorithms presented in:

> **Minkowski-Type Wasserstein Metrics and Barycenters for Location-Scale Mixtures with Application to Domain Adaptation**  
> Songyan Luo and Yunxin Zhang  
> *arXiv preprint [arXiv:2606.26482](https://arxiv.org/abs/2606.26482) (2026).*

---

## Data Preparation

All dataset files should be downloaded and placed under the `data/` directory.

- **VisDA Benchmark**: The extracted features for VisDA can be accessed via the Google Drive repository provided in [`eddardd/gmm-otda`](https://github.com/eddardd/gmm-otda).
- Please ensure that all extracted representations and dataset splits are correctly placed into the corresponding subdirectories inside `data/` prior to execution.

---

## Quick Start & Reproducibility

### 1. Basic Execution
To run the baseline pipeline in the background with deterministic seed initialization:

```bash
export PYTHONHASHSEED=42
nohup python3 run.py > run.log 2>&1 &
```

### 2. Full Benchmark Reproduction
The baseline execution demonstrates a representative experimental run. Reproducing the complete set of numerical and empirical results reported in the paper across different tasks and datasets requires setting the corresponding hyperparameters (e.g., mixture component numbers, regularization weights, and optimization options) as outlined in the paper and the accompanying configuration scripts.

---

## Acknowledgments & Attribution

This project is built upon and adapted from the open-source implementation:
- [**gmm-otda**](https://github.com/eddardd/gmm-otda): *Optimal Transport for Domain Adaptation through Gaussian Mixture Models* (TMLR 2025), along with its referenced upstream repositories.

We gratefully acknowledge the original authors for their open-source contributions.

---

## Citation

If you find this codebase or the theoretical framework useful in your research, please cite:

```bibtex
@article{luo2026minkowski,
  title   = {Minkowski-Type Wasserstein Metrics and Barycenters for Location-Scale Mixtures with Application to Domain Adaptation},
  author  = {Luo, Songyan and Zhang, Yunxin},
  journal = {arXiv preprint arXiv:2606.26482},
  year    = {2026}
}
```
