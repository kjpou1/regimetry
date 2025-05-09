# Positional Encoding in Regimetry

This document explains how **positional encoding** is integrated into the `regimetry` project to support temporal awareness in windowed market data.

## Why Positional Encoding?

In our regime detection pipeline, we generate rolling windows of shape `(num_windows, window_size, num_features)`. However, many neural models (like Transformers or Temporal CNNs) require **position-aware input** to model sequences effectively. Positional encodings inject information about the relative or absolute position of data within each window.

---

## Supported Encoding Methods

We support two types of encodings:

### 1. Sinusoidal (Fixed)

A deterministic, non-trainable encoding that encodes positions using sine and cosine functions.

* Based on the original Transformer paper.
* Adds alternating sine/cosine values across feature dimensions.
* Useful for consistent behavior across inference and training.

```python
X_pe = PositionalEncoding.add(X, method='sinusoidal')
```

### 2. Learnable

A trainable embedding layer that learns a positional representation for each timestep within a window.

* Each of the `window_size` positions gets a trainable vector of shape `(num_features,)`
* Embedding shape: `(window_size, num_features)`
* Applied as:

```python
X_pe = PositionalEncoding.add(X, method='learnable', learnable_dim=1273)
```

---

## Key Assumptions

* Inputs `X` must be shaped as `(batch_size, seq_len, d_model)`.
* In our case:

  * `batch_size = num_windows`
  * `seq_len = window_size` (e.g. 30)
  * `d_model = num_features` (e.g. 1273 after transformation)

---

## Broadcasting Explained

Both encoding types are added via broadcasting:

* Positional encodings are shaped `(1, seq_len, d_model)`
* They broadcast across the batch axis to match input shape

We removed all `embedding_dim` or `fixed_d_model` logic because the encoding is always reshaped to match the actual `d_model` from `X`.

---

## Learnable vs Sinusoidal: Which to Use?

| Use Case                   | Recommended Encoding |
| -------------------------- | -------------------- |
| Static rule-based models   | Sinusoidal           |
| Transformer-based models   | Learnable            |
| Consistent cross-inference | Sinusoidal           |
| Trainable feature dynamics | Learnable            |

---

## Related File

* Usage: `PositionalEncoding.add(X, method='sinusoidal' or 'learnable', learnable_dim=...)`

---

## Visualization Tip

You can visualize the encodings with:

```python
import matplotlib.pyplot as plt
pe = PositionalEncoding.get_sinusoidal_encoding(30, 1273)
plt.imshow(pe[:, :128].numpy(), cmap='viridis')
plt.colorbar()
plt.title("Sinusoidal Positional Encoding")
```

---
