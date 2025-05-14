
# 🚀 Getting Started

This guide walks you through running `regimetry` end-to-end using the included EUR/USD dataset. You’ll go from raw input → embeddings → clustering → interactive visualization in just a few steps.

---

## 📦 1. Clone the Repository

```bash
git clone https://github.com/kjpou1/regimetry.git
cd regimetry
```

---

## 🔧 2. Install Dependencies

> ✅ **Recommended:** Use [Poetry](https://python-poetry.org/) for environment and dependency management.

```bash
poetry install
```

If not using Poetry, install manually via pip:

```bash
pip install -r requirements.txt
```

---

## 📥 3. Ingest and Transform the Data

This will parse and clean the input data into a windowed feature format:

```bash
python run.py ingest \
  --signal-input-path examples/EUR_USD_processed_signals.csv
```

---

## 🔐 4. Generate Embeddings

Run the Transformer embedding pipeline over rolling windows:

```bash
python run.py embed \
  --signal-input-path examples/EUR_USD_processed_signals.csv \
  --output-name EUR_USD_embeddings.npy \
  --window-size 30 \
  --stride 1
```

> ⚠️ `window_size` must be smaller than the number of rows in your dataset.

---

## 🔗 5. Cluster the Embeddings

Apply spectral clustering and visualize regime patterns:

```bash
python run.py cluster \
  --embedding-path embeddings/EUR_USD_embeddings.npy \
  --regime-data-path data/processed/regime_input.csv \
  --output-dir reports/EUR_USD \
  --window-size 30 \
  --n-clusters 3
```

This will generate:

* `cluster_assignments.csv`
* t-SNE / UMAP plots
* Timeline chart
* Close price overlay
* Interactive HTML visualizations (if enabled)

---

## 🖼️ 6. Launch the Interactive Dashboard (Optional)

```bash
poetry run python -m dash_app.app
```

Then open in your browser:
👉 [http://localhost:8050](http://localhost:8050)

---

### 🧩 Dashboard Features

* **📉 Price Overlay** — view how cluster labels align with price movement.
* **🌀 t-SNE / 🔮 UMAP** — embedding projections by cluster.
* **🎨 Palette Preview** — view the current color palette and cluster mappings.

---

### ⚠️ Note on Config Upload

You can upload a `YAML` config (e.g., `configs/full_config.yaml`) to preview parameters like:

* `report_palette`
* `report_format`
* `window_size`
* `output_dir`

> 🛈 This is for **informational preview only**.
> To apply new config settings to the plots, rerun the `cluster` pipeline from CLI.
