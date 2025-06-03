# 📘 Regimetry Pipeline Overview

Regimetry provides tools for analyzing, clustering, and forecasting market regimes, combining transformer-based embeddings with spectral clustering to support strategy development and diagnostics.

This document outlines the full operational pipeline — from exploratory clustering to inference deployment — with phase objectives, expected outputs, and ideal operator roles.

---

## 📦 Components Overview

Regimetry is composed of the following major building blocks:

- **Embedding Encoder**:  
  `UnsupervisedTransformerEncoder` produces fixed-length embeddings over rolling windows of multivariate financial time series.

- **Clustering Layer**:  
  Spectral Clustering is used to segment the embedding space into interpretable regime clusters.

- **Forecast Model**:  
  Configurable forecasters (e.g., `stratum`, `stratum_attn`, `stratum_hydra`) predict the next embedding vector `Ê[t+1]`.

- **Classifier**:  
  A KNN model assigns predicted embeddings to the most probable regime cluster (`Cluster_ID[t+1]`).

- **Diagnostics & Reporting**:  
  Markov matrices, entropy, stickiness, and visual overlays (e.g., t-SNE, UMAP, Price) help interpret both regime structure and transitions.

---

## 🛠 How to Use This Document

This pipeline blueprint serves as a living reference for developing, evaluating, and deploying the Regimetry system. Use it to:

- **Track phase completion** and implementation status
- **Understand system boundaries** and data flows between modules
- **Assign responsibilities** via the *Ideal Operator* role hints
- **Identify backlog items** (e.g., planned drift detection, training profile snapshots)
- **Onboard new collaborators** or contributors to the system structure

Each phase is self-contained and designed to evolve independently — from research and tuning through to full production deployment.

---


## 🧭 Regimetry Pipeline Blueprint: From Regime Discovery to Predictive Deployment

---

### 📌 Phase 1: Cluster Configuration Search  
**"Analyze different configurations to determine the best cluster separation and distribution parameters to use."**

**Purpose**  
Systematically explore embedding and clustering parameters to identify the most meaningful and stable market regimes.

**Tasks**
- Search over:
  - `window_size`, `stride`, `embedding_dim`
  - `encoding_method` (learnable / sinusoidal)
  - `encoding_style` (stacked / interleaved)
  - `n_clusters`
- Assess:
  - Visual regime separation (t-SNE, UMAP, Price Overlay)
  - Quantitative structure (Markov matrix, entropy, stickiness)
  - Distribution balance

**Output**
- Ranked configuration shortlist
- Visual and metric-based reports
- `best_config.yaml`

**🎯 Ideal Operator**: *The Research Engineer*

---

### 📌 Phase 2: Canonical Profile & Periodic Regime Refresh  
**"Once we decide... this will be a profile we run periodically."**

**Purpose**  
Establish a fixed regime profile and rerun it regularly to refresh the current regime map.

**Tasks**
- Lock configuration from Phase 1
- Embed + cluster on schedule (daily, weekly, etc.)
- Monitor regime transitions
- Optionally forecast next regime

**Output**
- `locked_profile.yaml`
- Periodic overlays and Markov summaries
- Transition logs

**🎯 Ideal Operator**: *The Systems Architect*

---

### 📌 Phase 3: Train Forecast & Classification Models  
**"We will need to train and evaluate a model as well as the corresponding classifier."**

**Purpose**  
Train supervised models to predict embedding trajectories and classify future regime states.

**Tasks**
- Build rolling window training dataset
- Train:
  - Embedding forecaster (`Ê[t+1]`)
  - Regime classifier (e.g., KNN)
- Evaluate:
  - Forecast loss (MSE / cosine)
  - Accuracy, F1, confusion matrix

**Output**
- `embedding_forecaster.keras`
- `knn_classifier.joblib`
- `training_summary.json`
- 🕗 *(Backlog)* `training_profile_used.yaml`

**🎯 Ideal Operator**: *The Model Optimization Specialist*

---

### 📌 Phase 4: Inference Layer Deployment  
**"Deploy to an inference layer which we are defining now."**

**Purpose**  
Serve current and future regime predictions from trained models in real-time or batch mode.

**Tasks**
- Input most recent window `E[t-n:t]`
- Predict `Ê[t+1]` and `Cluster_ID[t+1]`
- Assign current cluster if needed
- Serve outputs to dashboards or strategies

**Output**
- `predict_next_cluster.py`
- Inference reports (CSV/JSON)
- Audit logs and integration hooks

**🎯 Ideal Operator**: *The MLOps Engineer*

---

### 📌 Phase 5: Drift & Transition Diagnostics (Future/Planned)  
**"Monitor regime behavior to detect unexpected transitions or breakdown in model alignment."**

**Purpose**  
Detect structural drift in regime behavior and model misalignment over time.

**Planned Tasks**
- Compare predicted vs actual transitions
- Monitor Markov matrix shifts and cluster instability
- Detect confidence drops and anomalies

**Planned Output**
- Drift logs and anomaly reports
- Transition deviation summaries

**🎯 Ideal Operator**: *The Watchtower Analyst*

