# regimetry

> **Mapping latent regimes in financial time series.**

![MIT License](https://img.shields.io/badge/license-MIT-green.svg)
![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)
![Built with TensorFlow](https://img.shields.io/badge/built%20with-TensorFlow-orange.svg?logo=tensorflow)
![Development Status](https://img.shields.io/badge/status-active-brightgreen)
![Made With ❤️](https://img.shields.io/badge/made%20with-%E2%9D%A4-red)

---
- [regimetry](#regimetry)
  - [📘 Overview](#-overview)
  - [🔍 What is a Regime?](#-what-is-a-regime)
  - [🧠 How It Works](#-how-it-works)
    - [1. **Data Ingestion**](#1-data-ingestion)
    - [2. **Embedding Pipeline**](#2-embedding-pipeline)
    - [3. **Clustering**](#3-clustering)
    - [4. **Visualization \& Interpretation**](#4-visualization--interpretation)
  - [📟 Command Line Usage](#-command-line-usage)
      - [🔹 Ingest Data](#-ingest-data)
      - [🔹 Generate Embeddings](#-generate-embeddings)
      - [🔹 Cluster Regimes](#-cluster-regimes)
  - [🧪 Example Dataset](#-example-dataset)
  - [🛠 Project Structure](#-project-structure)
  - [🧭 Orientation Going Forward](#-orientation-going-forward)
  - [✅ Status](#-status)
  - [🔗 Related Projects](#-related-projects)
  - [📖 Further Reading](#-further-reading)
  - [📄 License](#-license)
  - [👤 Author](#-author)
---
## 📘 Overview

**regimetry** is a *research-driven*, unsupervised regime detection engine for financial markets. It extracts latent structure from time-series data using deep learning embeddings and clustering, enabling traders and researchers to identify distinct behavioral phases in market activity.

This project begins as a learning ground and may evolve into a core component of my trading strategy pipelines.

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
- Features are typically sourced from [`ConvolutionLab`](https://github.com/kjpou1/ConvolutionLab),  
  but `regimetry` is **not dependent** on that specific pipeline — any compatible feature set can be used.  
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

## 📟 Command Line Usage


Run `regimetry` pipelines directly from the command line with optional overrides.

#### 🔹 Ingest Data

```bash
python run.py ingest \
  --signal-input-path examples/EUR_USD_processed_signals.csv
```

This will:

* Parse the input CSV
* Normalize and structure features
* Save the result to `artifacts/data/processed/`

#### 🔹 Generate Embeddings

```bash
python run.py embed \
  --signal-input-path examples/EUR_USD_processed_signals.csv \
  --output-name EUR_USD_embeddings.npy
```

This will:

* Apply rolling window and positional encoding
* Run transformer encoder to extract dense regime embeddings
* Save the result to `artifacts/embeddings/EUR_USD_embeddings.npy`

> If `--output-name` is not provided, the default file is `embeddings.npy`.

#### 🔹 Cluster Regimes

```bash
python run.py cluster \
  --embedding-path embeddings/EUR_USD_embeddings.npy \
  --regime-data-path data/processed/regime_input.csv \
  --output-dir reports/EUR_USD \
  --window-size 30 \
  --n-clusters 3
```

This will:

* Load precomputed transformer embeddings
* Run Spectral Clustering to assign regime IDs
* Generate visualizations (t-SNE, UMAP, timeline, chart overlays)
* Save results to the specified output directory

> You can also run this with a config file:
>
> ```bash
> python run.py cluster --config configs/cluster_config.yaml
> ```


---

## 🧪 Example Dataset

An example file is included at [`examples/EUR_USD_processed_signals.csv`](examples/EUR_USD_processed_signals.csv) to help you test the pipeline immediately.

This file contains:
- Processed technical indicators (AHMA, LP, LC, ATR, etc.)
- Cleaned and aligned daily bars for EUR/USD
- A ready-to-ingest format compatible with the full `embedding_pipeline`

You can run the **ingestion pipeline** on this dataset:

```bash
python run.py ingest --signal-input-path examples/EUR_USD_processed_signals.csv
````

— OR —

Run the **embedding pipeline** to generate transformer embeddings:

```bash
python run.py embed --signal-input-path examples/EUR_USD_processed_signals.csv
```

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
```

---

## 🧭 Orientation Going Forward

* Start with regime labeling and visualization
* Build diagnostic tools to analyze regime behavior
* Eventually tie `regime_id` into strategy filters and signal validation
* Keep architecture modular, interpretable, and ready for real-world integration

---

## ✅ Status

* [x] Core concept defined
* [x] Data ingestion pipeline implemented
* [x] Transformer encoder + positional encoding embedded
* [x] Embedding pipeline operational and CLI-integrated
* [x] Embeddings saved to `embeddings/`
* [x] Spectral clustering and regime ID assignment
* [x] Visualization tools (UMAP, t-SNE) with cluster overlay
* [x] Historical regime labeling and export
* [ ] Live inference support
* [ ] Contrastive or autoregressive pretraining options

---

## 🔗 Related Projects

- [`ConvolutionLab`](https://github.com/kjpou1/ConvolutionLab):  
  A technical feature engineering framework that produces structured indicators (e.g., AHMA, LP, LC, ATR)  
  used as inputs to `regimetry`.  
  **Note:** While `regimetry` is compatible with ConvolutionLab outputs, it is not tightly coupled to it —  
  any feature-rich dataset with proper formatting can be used for embedding and clustering.

---

## 📖 Further Reading  
For foundational papers, models, and tools behind the `regimetry` pipeline, see the [References](./docs/REFERENCES_README.md).

---

## 📄 License

MIT

## 👤 Author

\kjpou1 — Initial maintainer
