# regimetry

> **Mapping latent regimes in financial time series.**

![MIT License](https://img.shields.io/badge/license-MIT-green.svg)
![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)
![Built with TensorFlow](https://img.shields.io/badge/built%20with-TensorFlow-orange.svg?logo=tensorflow)
![Visualize with Dash](https://img.shields.io/badge/Dash-Framework-blue?logo=dash)
![Managed with Poetry](https://img.shields.io/badge/Poetry-Dependency--Mgmt-purple)
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
  - [🚀 Getting Started](#-getting-started)
    - [📘 Regime Detection Window Delay](#-regime-detection-window-delay)
  - [📟 Command Line Usage](#-command-line-usage)
      - [🔹 Ingest Data](#-ingest-data)
    - [🔹 Generate Embeddings](#-generate-embeddings)
      - [🛠 Available CLI Arguments for `embed`](#-available-cli-arguments-for-embed)
    - [🔹 Cluster Regimes](#-cluster-regimes)
      - [🛠 Available CLI Arguments for `cluster`](#-available-cli-arguments-for-cluster)
  - [🧪 Example Dataset](#-example-dataset)
  - [🛠️ Configuration Files](#️-configuration-files)
    - [📂 Example Config](#-example-config)
  - [✅ Section: Configuration Files → Example Config](#-section-configuration-files--example-config)
    - [🧠 Usage in CLI](#-usage-in-cli)
    - [🖼️ Usage in Dash App](#️-usage-in-dash-app)
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

See the full step-by-step guide:
📖 [`docs/GETTING_STARTED_README.md`](docs/GETTING_STARTED_README.md)

> Includes:
>
> * Git clone instructions
> * Poetry or manual install
> * Data ingestion
> * Embedding generation
> * Regime clustering
> * Optional Dash dashboard launch

---

### 📘 Regime Detection Window Delay

> 📄 See: [`docs/REGIME_DETECTION_README.md`](docs/REGIME_DETECTION_README.md)

Because regime labels are assigned based on **rolling windows**, the cluster ID for the final bars of a dataset **cannot be known until the full window is complete**.

For example, with a `window_size = 30`:

* The first 29 bars will not receive a regime ID
* The **last 29 bars** also **do not reflect any future regime change**, since there are no forward windows to reclassify them

This introduces a **natural lag** in regime detection:

* New regimes will only appear after enough time has passed for the model to “observe” a full window in the new market condition.

👉 For more details, see the full explanation: [`REGIME_DETECTION_README.md`](docs/REGIME_DETECTION_README.md)

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

Ah — got it. Since `--embedding-dim` is now **used for both `learnable` and `sinusoidal`**, the description needs to be updated accordingly. Here's the revised table and footnote:

---

#### 🛠 Available CLI Arguments for `embed`

| Argument              | Description                                                                     |
| --------------------- | ------------------------------------------------------------------------------- |
| `--signal-input-path` | Path to the CSV file with feature-enriched signal data                          |
| `--output-name`       | Optional output file name for the `.npy` embeddings (default: `embeddings.npy`) |
| `--window-size`       | Number of time steps per rolling window (default: `30`)                         |
| `--stride`            | Step size between rolling windows (default: `1`)                                |
| `--encoding-method`   | Positional encoding method: `sinusoidal` (default) or `learnable`               |
| `--encoding-style`    | Sinusoidal encoding format: `interleaved` (default) or `stacked`                |
| `--embedding-dim`     | Embedding dimension to use for both sinusoidal and learnable encodings          |
| `--config`            | Optional YAML config path to override pipeline settings                         |
| `--debug`             | Enable debug logging                                                            |

> ℹ️ **Note:** `--embedding-dim` applies to **both** `sinusoidal` and `learnable` encodings.
> For `sinusoidal`, it sets the generated frequency embedding size. For `learnable`, it defines the trainable positional embedding dimension.


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

## 🛠️ Configuration Files

`regimetry` supports YAML configuration files to streamline pipeline execution and visualization setup. These configs centralize all key parameters used by the CLI and Dash dashboard.

### 📂 Example Config

Here's a fully annotated example config file at [`config/full_config.yaml`](config/full_config.yaml):

```yaml
# ✅ General Settings
debug: true

# ✅ Ingestion Settings
signal_input_path: ./examples/EUR_USD_processed_signals.csv
include_columns: "*"
exclude_columns: ["Date", "Hour"]  # Remove Date/Hour for daily resolution

deterministic: true                # Enables reproducible embeddings and clustering
random_seed: 42                    # Controls randomness for TF, t-SNE, UMAP, Spectral Clustering

# ✅ Embedding Settings
output_name: EUR_USD_embeddings.npy
window_size: 10
stride: 1
encoding_method: "sinusoidal"      # Options: 'sinusoidal', 'learnable'
encoding_style: "interleaved"      # Options: 'interleaved', 'stacked'
# embedding_dim: 80

# ✅ Clustering Settings
embedding_path: ./embeddings/EUR_USD_embeddings.npy
regime_data_path: ./data/processed/regime_input.csv
output_dir: ./reports/EUR_USD
n_clusters: 8

# ✅ Report Settings
report_format: ["matplotlib", "plotly"]  # Options: [], ["matplotlib"], ["plotly"]
report_palette: Set2                     # Any valid seaborn palette name
```

Your `README.md` is already outstanding — clean, modular, and informative. To reflect your recent changes, here’s a **drop-in-ready update section** you can patch under:

---

## ✅ Section: Configuration Files → Example Config

Update the example YAML to include the new deterministic settings:

```yaml
# ✅ Embedding Settings
output_name: EUR_USD_embeddings.npy
window_size: 10
stride: 1
encoding_method: "learnable"       # Options: 'sinusoidal', 'learnable'
encoding_style: "interleaved"      # Used only for 'sinusoidal'

embedding_dim: 71                  # Required for 'learnable'; optional for 'sinusoidal'
deterministic: true                # Enables reproducible embeddings and clustering
random_seed: 42                    # Controls randomness for TF, t-SNE, UMAP, Spectral Clustering
```

> 🧬 **Determinism Note:**  
> When `deterministic: true`, all randomness (including Transformer, t-SNE, UMAP, Spectral Clustering) is locked using `random_seed`.  
> This ensures identical results across re-runs with the same input data.  
> When `false`, variability is allowed — useful for exploration or stress testing.  
>  
> 🔗 [Learn more → Reproducibility Controls](./docs/REPRODUCIBILITY_README.md)

---

### 🧠 Usage in CLI

You can run any pipeline stage using a config override:

```bash
python run.py cluster --config config/full_config.yaml
```

* CLI will auto-resolve relative paths (e.g., to `./data/`, `./embeddings/`)
* Config values override internal defaults
* Any CLI argument passed explicitly will override the config

> ✅ **CLI flags always take precedence** over values defined in the YAML.

---

### 🖼️ Usage in Dash App

The Dash dashboard can also load and preview a YAML config:

```bash
poetry run python -m dash_app.app
```

In the **Palette Preview** tab:

* Upload any `.yaml` file
* The dashboard will display:

  * Parsed settings (`window_size`, `n_clusters`, `output_dir`, etc.)
  * Current seaborn `report_palette` rendered as a color swatch

> ⚠️ *This is for preview only — uploaded config **does not affect** the rendered plots.*
> To change plots, rerun the `cluster` CLI with the updated config.

---

For a full reference of all supported fields, see:
📘 [`docs/CONFIG_REFERENCE_README.md`](docs/CONFIG_REFERENCE_README.md)

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

---

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
