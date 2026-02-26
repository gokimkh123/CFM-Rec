# ColdStart-FlowCF-TF
**Flow Matching based Cold-Start Recommendation System**

This repository implements a **Continuous-Time Flow Matching** model for recommender systems, specifically designed to address the **Cold-Start Problem**.

By utilizing **User Activity Priors (Bernoulli/Gaussian distributions)** instead of pure Gaussian noise as the initial state ($x_0$), this model effectively infers the preference distribution of new items (Cold Items) using only their side information (e.g., tags, genres).

---

## 📂 1. Directory Structure

```text
ColdStart-FlowCF-TF/
├── data/                     # Dataset directory
│   ├── ML1M/                 # MovieLens-1M raw data
│   └── side_info.npy         # Pre-processed Item Side Information (Embeddings)
├── src/                      # Source code
│   ├── __init__.py
│   ├── data_loader.py        # Data pipeline with Cold-Start masking strategy
│   ├── model.py              # FlowModel architecture (MLP + Time Embeddings)
│   ├── flow_logic.py         # Flow Matching logic (Vector Field & ODE Solver)
│   └── metrics.py            # Evaluation metrics (Recall@K, NDCG@K)
├── config.yaml      # Hyperparameter configurations
├── train.py                  # Main training loop with tf.GradientTape
├── evaluate.py               # Evaluation script for cold-start scenarios
├── run_all.py                # Automated experiment script (Step ablation study)
├── inference.py              # Single item inference script
├── Dockerfile                # Docker build configuration
├── docker-compose.yml        # Container orchestration and port settings
└── requirements.txt          # Python dependencies
```

## 🐳 2. Environment Setup (Docker)
This project is optimized for TensorFlow GPU environments. We strongly recommend using Docker and Docker Compose for a seamless setup.

Prerequisites
Docker & Docker Compose


Quick Start
Build and run the container using docker compose. This handles port forwarding (6006) and volume mounting automatically.

```Bash
# 1. Build and start the container in background
docker compose up -d --build

# 2. Access the container shell
docker exec -it cold_flow_tf /bin/bash
```
Note: The docker-compose.yml mounts the current directory to /app. Any code changes made locally will be immediately reflected inside the container.


## 🚀 3. Usage
Run the following commands inside the Docker container.

3.1. Training (Single Run)
Train the model with a specific number of Euler steps.


```Bash
# Train with default 10 steps
python train.py --steps 10
```
--steps: Defines both training time-discretization ($n\_step$) and inference sampling steps.

## 📊 4. Monitoring (TensorBoard)
Real-time monitoring of Loss curves and Recall@20 performance.

Launch TensorBoard inside the container:
```Bash
tensorboard --logdir logs_{dataset}/ --port 6006 --bind_all
```
Access from your local machine:

Open your browser and visit: http://localhost:6006

VS Code Users: Go to the PORTS tab (bottom panel) and click the globe icon (🌐) next to port 6006.


