# OTDA-LSMM

[![arXiv](https://img.shields.io/badge/arXiv-2606.26482-b31b1b.svg)](https://arxiv.org/abs/2606.26482)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

This repository provides the official implementation of the algorithms presented in:

> **Minkowski-Type Wasserstein Metrics and Barycenters for Location-Scale Mixtures with Application to Domain Adaptation**  
> Songyan Luo and Yunxin Zhang  
> *arXiv preprint [arXiv:2606.26482](https://arxiv.org/abs/2606.26482) (2026).*

> **Note:** The initial arXiv preprint ([arXiv:2606.26482v1](https://arxiv.org/abs/2606.26482)) contains typographical errors. Please refer to subsequent revisions.


## Data Preparation

All dataset files should be downloaded and placed under the `data/` directory.

- **VisDA Benchmark**: The extracted features for VisDA can be accessed via the Google Drive link provided in [`eddardd/gmm-otda`](https://github.com/eddardd/gmm-otda).


## Quick Start

By default, running `run.py` reproduces the **best accuracy** reported in the paper:

```bash
export PYTHONHASHSEED=42
nohup python3 run.py > run.log 2>&1 &
```

To run experiments with other settings, modify the command-line arguments accordingly (e.g., `--features="resnet101"`).


## Acknowledgments & Attribution

This project is built upon and adapted from the open-source implementation:
- [**gmm-otda**](https://github.com/eddardd/gmm-otda): *Optimal Transport for Domain Adaptation through Gaussian Mixture Models* (TMLR 2025), along with its references.

We gratefully acknowledge the open-source contributions of the original authors.


## Citation

If you find this codebase useful, please cite:

```bibtex
@article{luo2026minkowskitypewassersteinmetricsbarycenters,
      title={Minkowski-Type Wasserstein Metrics and Barycenters for Location-Scale Mixtures with Application to Domain Adaptation}, 
      author={Songyan Luo and Yunxin Zhang},
      year={2026},
      eprint={2606.26482},
      archivePrefix={arXiv},
      primaryClass={math.OC},
      url={https://arxiv.org/abs/2606.26482}, 
}
```
