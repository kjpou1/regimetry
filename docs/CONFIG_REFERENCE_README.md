
# 📄 Config Reference — `regimetry`

This document lists all configurable keys supported by the `full_config.yaml` file in the `regimetry` project. These values can be overridden via:

* Command-line flags (`--config path/to/config.yaml`)
* Environment variables (where applicable)
* The default file: `config/full_config.yaml`

---

## ✅ General Settings

| Key     | Type   | Description                              | Default |
| ------- | ------ | ---------------------------------------- | ------- |
| `debug` | `bool` | Enables verbose logging and debug output | `false` |

---

## 🧩 Ingestion Settings

| Key                 | Type         | Description                                        | Example                                    |
| ------------------- | ------------ | -------------------------------------------------- | ------------------------------------------ |
| `signal_input_path` | `str`        | Path to input CSV with signal-enriched time series | `./examples/EUR_USD_processed_signals.csv` |
| `include_columns`   | `list`/`str` | Columns to include. `"*"` means all columns        | `["Close", "AHMA", "LC_Slope"]` or `"*"`   |
| `exclude_columns`   | `list`       | Columns to exclude (e.g., date/time columns)       | `["Date", "Hour"]`                         |

---

## 🔐 Embedding Settings

| Key               | Type  | Description                                                           | Default          |
| ----------------- | ----- | --------------------------------------------------------------------- | ---------------- |
| `output_name`     | `str` | Filename for generated embeddings (`.npy`)                            | `embeddings.npy` |
| `window_size`     | `int` | Number of bars per rolling window                                     | `30`             |
| `stride`          | `int` | Rolling window stride                                                 | `1`              |
| `encoding_method` | `str` | Positional encoding method: `sinusoidal` or `learnable`               | `sinusoidal`     |
| `encoding_style`  | `str` | Layout for encoding: `interleaved` or `stacked`                       | `interleaved`    |
| `embedding_dim`   | `int` | Required if `encoding_method: learnable`. Sets learnable encoding dim | *None*           |

---

## 🔗 Clustering Settings

| Key                | Type  | Description                              | Example                               |
| ------------------ | ----- | ---------------------------------------- | ------------------------------------- |
| `embedding_path`   | `str` | Path to `.npy` file with embeddings      | `./embeddings/EUR_USD_embeddings.npy` |
| `regime_data_path` | `str` | Path to the processed regime-aligned CSV | `./data/processed/regime_input.csv`   |
| `output_dir`       | `str` | Output directory for clustering results  | `./reports/EUR_USD`                   |
| `n_clusters`       | `int` | Number of regimes (clusters) to generate | `3`                                   |

---

## 📊 Report Settings

| Key              | Type        | Description                                                                    | Example                    |
| ---------------- | ----------- | ------------------------------------------------------------------------------ | -------------------------- |
| `report_format`  | `list[str]` | Controls which plot types to generate: `["matplotlib"]`, `["plotly"]`, or both | `["matplotlib", "plotly"]` |
| `report_palette` | `str`       | Seaborn-compatible palette name (e.g., `Set2`, `tab10`, `Dark2`, `husl`)       | `Set2`                     |

> ⚠️ Note: `report_palette` affects color consistency across Matplotlib + Plotly.
> Plot visuals are fixed once generated — config changes only affect future runs.

---

## Example Full Config

```yaml
debug: true
signal_input_path: ./examples/EUR_USD_processed_signals.csv
include_columns: "*"
exclude_columns: ["Date", "Hour"]

output_name: EUR_USD_embeddings.npy
window_size: 10
stride: 1

encoding_method: "sinusoidal"
encoding_style: "interleaved"

embedding_path: ./embeddings/EUR_USD_embeddings.npy
regime_data_path: ./data/processed/regime_input.csv
output_dir: ./reports/EUR_USD
n_clusters: 8

report_format: ["matplotlib", "plotly"]
report_palette: Set2
```

