# FlowCF: Item Cold-Start Implementation

This repository is an **unofficial modified version** of the paper *"Flow Matching for Collaborative Filtering (KDD 2025)"*.

### Original Source
* **Paper:** [Flow Matching for Collaborative Filtering (arXiv)](https://arxiv.org/abs/2502.07303)
* **Original Repository:** [[Insert Link to Original GitHub Repo Here](https://github.com/chengkai-liu/FlowCF)]

### Key Modifications
This project adapts the original FlowCF model specifically for **Item Cold-Start scenarios**.
The code has been modified to:

1.  **Simulate Cold-Start:** Predict potential target users for newly released items (based on seed interactions).
2.  **Dockerization:** Provide a stable Docker environment for reproducibility.
3.  **Optimization:** Tuned hyperparameters for high-performance training on Linux servers.

---
*Note: This code is for educational and research purposes only. All rights to the original logic belong to the authors.*
## Quick Start

### 1. Prerequisites
* Docker & Docker Compose
* NVIDIA Drivers (for GPU support)

### 2. Build the Environment
```bash
docker compose build
```

### 3. Train the Model
This will start training using the optimized configuration in flowcf.yaml.

```bash
docker compose up
```
The trained model checkpoint will be saved in the saved/ directory (e.g., saved/FlowCF-Jan-12-xxxx.pth).


### 4. Inference (Simulation) & Evaluate Performance
Simulate a "New Movie Release" scenario using specific seed users.
Test the model's accuracy on Item Cold-Start scenarios (Recall@K, NDCG@K).
```bash
# Replace the filename with your actual saved model
docker compose run --rm flowcf python evaluate.py --checkpoint saved/FlowCF-Jan-12-2026_09-39-32.pth

docker compose run --rm flowcf python inference.py --checkpoint saved/FlowCF-Jan-12-2026_09-39-32.pth --mid 100
```

## Configuration (flowcf.yaml)
Key hyperparameters have been tuned for better convergence and speed:

- train_batch_size: Increased to 1024 (Balanced for detail & speed).
- eval_batch_size: set to 40960 for rapid evaluation.
- s_steps: Set to 10 for more precise ODE solver inference.
- dims_mlp: Tuned to [1000, 1000, 1000] for optimal capacity.
