# 📄 Reproducibility in `regimetry`

`regimetry` includes explicit controls to ensure that all embedding, clustering, and visualization results can be reproduced exactly when desired. This is critical for both scientific integrity and trading reliability.

---

## 🔒 Reproducibility Controls

To guarantee deterministic behavior:

### ✅ `deterministic: true`

Enable this in your config to activate full reproducibility across:
- TensorFlow (model weight init, dropout)
- NumPy (data transformations)
- Python's `random` module
- Spectral Clustering
- t-SNE and UMAP

### ✅ `random_seed: 42`

Set a fixed seed that will be used globally throughout the embedding and clustering pipeline.

```yaml
deterministic: true
random_seed: 42
````

This guarantees that:

* Same input data
* Same config
* Same code version
  ➡️ will always produce **identical embeddings**, **cluster assignments**, and **PDF reports**.

---

## 🎯 Behavior Summary

| Config Setting         | Output Behavior               |
| ---------------------- | ----------------------------- |
| `deterministic: true`  | Identical results on rerun    |
| `random_seed: 42`      | All randomness controlled     |
| `deterministic: false` | Allows stochastic variation   |
| `random_seed: 1337`    | Different, but stable results |

---

## 🔧 Implementation

These controls are enforced at runtime:

```python
if config.deterministic:
    set_deterministic(seed=config.random_seed)
```

And passed to:

* `SpectralClustering(random_state=seed)`
* `TSNE(random_state=seed)`
* `UMAP(random_state=seed)`
* `tf.random.set_seed(seed)`

---

## 🧪 How to Validate

You can validate reproducibility with:

```python
np.array_equal(embeddings_v1, embeddings_v2)  # Should be True
text_v1 == text_v2  # Extracted from PDF report
pd.read_csv("clusters_v1.csv").equals(pd.read_csv("clusters_v2.csv"))
```

You should see:

* `max_abs_diff = 0.0` between embeddings
* Identical PDF cluster overlays and prompts

---

## ⚠️ Common Pitfalls

* ❌ Forgetting to set `deterministic: true` will break reproducibility
* ❌ Changing `window_size` or features between runs will cause misalignment
* ❌ Different `random_seed` values will create deterministic *but distinct* outputs

---

## ✅ Best Practices

* Always version your config files (e.g. `full_config.yaml`)
* Save final YAML config used per run (e.g. `config_resolved.yaml`)
* Include experiment ID or hash in output filenames
* Use reproducibility controls when clustering for production or publication

---

## 📁 Example

```yaml
# config/full_config.yaml
deterministic: true
random_seed: 42
window_size: 5
stride: 1
encoding_method: "learnable"
embedding_dim: 71
n_clusters: 10
```

---

## 📎 Related Files

* [`config/full_config.yaml`](../config/full_config.yaml)
* [`ClusteringReportService`](../src/regimetry/services/clustering_report_service.py)

```
