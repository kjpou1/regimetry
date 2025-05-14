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
  - [� Getting Started](#-getting-started)
    - [1. 🔧 Install Dependencies](#1--install-dependencies)
    - [2. 📥 Ingest + Transform Data](#2--ingest--transform-data)
    - [3. 🔐 Generate Embeddings](#3--generate-embeddings)
    - [4. 🔗 Cluster the Embeddings](#4--cluster-the-embeddings)
    - [5. 🖼️ Launch the Interactive Dashboard (Optional)](#5-️-launch-the-interactive-dashboard-optional)
  - [📟 Command Line Usage](#-command-line-usage)
      - [🔹 Ingest Data](#-ingest-data)
    - [🔹 Generate Embeddings](#-generate-embeddings)
      - [🛠 Available CLI Arguments for `embed`](#-available-cli-arguments-for-embed)
    - [🔹 Cluster Regimes](#-cluster-regimes)
      - [🛠 Available CLI Arguments for `cluster`](#-available-cli-arguments-for-cluster)
  - [🧪 Example Dataset](#-example-dataset)
  - [🖥️ Interactive Dashboard](#️-interactive-dashboard)
    - [🚀 Launch the App](#-launch-the-app)
    - [🧩 Features](#-features)
    - [📂 Directory Structure](#-directory-structure)
    - [📦 Example Config for Palette Preview](#-example-config-for-palette-preview)
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

## 🚀 Getting Started

This section gives you the fastest way to test `regimetry` end-to-end on the included EUR/USD dataset.

### 1. 🔧 Install Dependencies

> Recommended: Use [Poetry](https://python-poetry.org/) to manage environments.

```bash
poetry install
```

Or manually:

```bash
pip install -r requirements.txt
```

---

### 2. 📥 Ingest + Transform Data

Use the built-in test file to generate processed input:

```bash
python run.py ingest \
  --signal-input-path examples/EUR_USD_processed_signals.csv
```

---

### 3. 🔐 Generate Embeddings

```bash
python run.py embed \
  --signal-input-path examples/EUR_USD_processed_signals.csv \
  --output-name EUR_USD_embeddings.npy \
  --window-size 30 \
  --stride 1
```

---

### 4. 🔗 Cluster the Embeddings

```bash
python run.py cluster \
  --embedding-path embeddings/EUR_USD_embeddings.npy \
  --regime-data-path data/processed/regime_input.csv \
  --output-dir reports/EUR_USD \
  --window-size 30 \
  --n-clusters 3
```

---

### 5. 🖼️ Launch the Interactive Dashboard (Optional)

```bash
poetry run python -m dash_app.app
```

Open in browser: [http://localhost:8050](http://localhost:8050)

> 🛈 Upload a YAML config (e.g., `configs/full_config.yaml`) to preview your settings and color palette.
> ⚠️ This config **does not change the plots** — it’s informational only.
> To change plot visuals, rerun the `cluster` pipeline with new settings.

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

### 🔹 Generate Embeddings

```bash
python run.py embed \
  --signal-input-path examples/EUR_USD_processed_signals.csv \
  --output-name EUR_USD_embeddings.npy \
  --window-size 30 \
  --stride 1 \
  --encoding-method sinusoidal \
  --encoding-style interleaved
```

This will:

* Apply a rolling window (default: 30 bars, stride: 1 unless overridden)
* Use positional encoding and Transformer to generate embeddings
* Save the result to `embeddings/EUR_USD_embeddings.npy`

> ⚠️ **Note:** Ensure that `window_size` is smaller than your dataset length.
> If `window_size >= len(data)`, no embeddings will be produced.

---

#### 🛠 Available CLI Arguments for `embed`

| Argument              | Description                                                                     |
| --------------------- | ------------------------------------------------------------------------------- |
| `--signal-input-path` | Path to the CSV file with feature-enriched signal data                          |
| `--output-name`       | Optional output file name for the `.npy` embeddings (default: `embeddings.npy`) |
| `--window-size`       | Number of time steps per rolling window (default: `30`)                         |
| `--stride`            | Step size between rolling windows (default: `1`)                                |
| `--encoding-method`   | `sinusoidal` (default) or `learnable`                                           |
| `--encoding-style`    | `interleaved` (default) or `stacked` (only used if method is `sinusoidal`)      |
| `--embedding-dim`     | Required if using `learnable` encoding (defines learnable position dimension)   |
| `--config`            | Optional YAML config path to override pipeline settings                         |
| `--debug`             | Enable debug logging                                                            |

> ⚠️ **Note:** `--embedding-dim` is **only used** with `--encoding-method=learnable`.
> If specified with `sinusoidal`, it will be ignored with a warning.



### 🔹 Cluster Regimes

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
* Apply spectral clustering to assign regime IDs
* Align cluster labels with original time-series data (using `window_size` for offset)
* Generate visualizations (t-SNE, UMAP, timeline, and price overlay)
* Save outputs to the specified report directory

> ⚠️ **Note:** The `window_size` used here **must match** the one used during embedding.
> Otherwise, the cluster labels will not align correctly with the input time series.

---

#### 🛠 Available CLI Arguments for `cluster`

| Argument             | Description                                                                    |
| -------------------- | ------------------------------------------------------------------------------ |
| `--embedding-path`   | Path to the `.npy` file with saved embeddings                                  |
| `--regime-data-path` | CSV file containing the signal-enriched time series (e.g., `regime_input.csv`) |
| `--output-dir`       | Directory to save visualizations and labeled data                              |
| `--window-size`      | Window size used during embedding (used for alignment)                         |
| `--n-clusters`       | Number of regimes (clusters) to detect (default: `3`)                          |
| `--config`           | Optional YAML config file to provide all arguments at once                     |
| `--debug`            | Enable debug logging                                                           |


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

## 🖥️ Interactive Dashboard

`regimetry` ships with an optional Dash app that provides a user-friendly interface for exploring clustering results.

### 🚀 Launch the App

```bash
poetry run python -m dash_app.app
```

The app will run locally at [http://localhost:8050](http://localhost:8050)

> ⚠️ Requires `dash`, `dash-bootstrap-components`, and `plotly` installed in your environment.

### 🧩 Features

* **📁 YAML Config Loader**
  Upload a YAML config file (e.g., `configs/full_config.yaml`) to view the current settings:

  * `window_size`

  * `report_palette`

  * `output_dir`

  * `report_format`

  > 🛈 *This is for **informational preview only** — uploading a config file does **not** affect the rendered plots.*
  > Plots are static and must be regenerated via the CLI (`run.py cluster`) if you want different parameters applied.

* **🧠 Cluster Visualizations**

  * `📉 Price Overlay`: Close price with color-coded cluster markers
  * `🌀 t-SNE`: 2D projection of regime embedding space
  * `🔮 UMAP`: Alternative manifold-based view of clusters

* **🎨 Palette Preview**

  * Auto-detects and displays the seaborn color palette in use
  * Ensures consistent cluster color mapping between matplotlib and Plotly
  * Preview updates when a new YAML config is uploaded

### 📂 Directory Structure

```bash
dash_app/
├── app.py               # Main Dash app with config reader and tab layout
├── ...
```

### 📦 Example Config for Palette Preview

```yaml
report_format: ["matplotlib", "plotly"]
report_palette: "Set2"
output_dir: ./artifacts/reports/EUR_USD
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
