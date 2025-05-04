# regimetry

> **Mapping latent regimes in financial time series.**

---

## 📘 Overview

**regimetry** is a *research-driven*, unsupervised regime detection engine for financial markets. It extracts latent structure from time-series data using deep learning embeddings and clustering, enabling traders and researchers to identify distinct behavioral phases in market activity.

This project begins as a learning ground and may evolve into a core component of production-grade strategy pipelines.

---

## 🔍 What is a Regime?

In `regimetry`, a **regime** is a *latent, temporally coherent pattern* in market behavior. It reflects distinct combinations of volatility, trend, momentum, and signal alignment — not defined by hand, but discovered through patterns in the data itself.

Formally:
- Regimes are clusters in the embedding space of rolling market windows (e.g., 30 bars).
- These clusters represent *behavioral states* the market tends to revisit.
- The system learns these states using unsupervised learning (e.g., transformer encoder + spectral clustering).

---

## 🧠 How It Works

### 1. **Data Ingestion**
- Load daily bar data per instrument  
- Normalize features (Close, AHMA, LP, LC, etc.)  
- Slice into overlapping windows (default: 30 bars, stride 1)

### 2. **Embedding Pipeline**
- **Pass each window through a Transformer encoder (default),** which maps the window into a dense latent representation.  
- This step is modular and can later be replaced with alternative architectures (e.g., autoencoders, SimCLR, CNNs).

### 3. **Clustering**
- Standardize the embeddings  
- Cluster them using Spectral Clustering (or another method)  
- Assign each window a `regime_id`

### 4. **Visualization & Interpretation**
- Use t-SNE or UMAP to project embeddings  
- Visualize regime transitions over time  
- Map regimes back to chart or signal data for strategy insights

---

## 🛠 Project Structure

```bash
regimetry/
├── models/               # Trained encoders and clustering artifacts
├── data/                 # Input raw / processed datasets
├── artifacts/            # JSON logs, regime labels, regime visual outputs
├── config.yaml           # Tunable pipeline settings
├── pyproject.toml
└── README.md
````

---

## 🧭 Orientation Going Forward

* Start with regime labeling and visualization
* Build diagnostic tools to analyze regime behavior
* Eventually tie `regime_id` into strategy filters and signal validation
* Keep architecture modular, interpretable, and ready for real-world integration

---

## ✅ Status

* [x] Core concept defined
* [ ] Transformer encoder training setup
* [ ] Clustering pipeline prototype
* [ ] Historical regime labeling and export
* [ ] Live regime inference module

---

## 📄 License

MIT

## 👤 Author

\kjpou1 — Initial maintainer
