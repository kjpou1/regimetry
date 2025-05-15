# 📊 Regime Detection Behavior

This document explains how regime assignments are computed, and why regime change detection is **delayed near the end of the dataset**.

---

## 🔁 Rolling Window Embedding

The regime clustering pipeline uses **rolling windows of fixed length** (configured via `window_size`) to generate transformer embeddings for each time point.

Each embedding represents the **market behavior across `window_size` bars**, not a single point in time.

### Example:  
With `window_size = 30`, the embedding at index `t = 59` is computed using the window:
```

\[30 bars] → indices 30 to 59

```

---

## 🧠 Regime Assignment Mechanics

After embeddings are computed, they are passed through:
- **Spectral Clustering** to assign discrete cluster labels
- **Dimensionality Reduction** (t-SNE / UMAP) for visualization

Cluster labels are then **aligned back to the original dataframe**, starting from index:
```

start = window\_size - 1

````

---

## ⏳ Delay in Detecting Regime Change

Because clustering requires **a complete window ending at each index**, **you cannot assign a regime to the final `window_size - 1` rows** of the dataset.

### ❗ Consequence:

> **Any new regime change occurring within the last `window_size - 1` bars will not be detected until future data completes those windows.**

---

## ✅ Timeline Summary

| Index Range                   | Regime Cluster (`Cluster_ID`) |
|-------------------------------|-------------------------------|
| `0` to `window_size - 2`      | ❌ Not available (NaN)         |
| `window_size - 1` to `N-1`    | ✅ Assigned via clustering     |
| `N` to `N + window_size - 2`  | ❌ Not available (future req.) |

---

## 🧩 Visualization Tip

To ensure visual continuity in the dashboard:

- Use `.ffill()` on `Cluster_ID` for plotting the trailing bars  
- Optionally mark the **last valid cluster index** with a vertical line:

```python
last_cluster_index = window_size - 1 + len(cluster_labels) - 1
plt.axvline(x=last_cluster_index, color='gray', linestyle='--')
````

---

## 📎 Related Configs

* `window_size` (int): Number of bars per embedding window
* `stride` (int): Step between windows (usually 1)
* `n_clusters` (int): Number of regime clusters to form

---

## 🛠️ Diagnostic Warnings

If the number of cluster labels does not match `len(df) - window_size + 1`, a diagnostic error will be raised during the clustering pipeline.

