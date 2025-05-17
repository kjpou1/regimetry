# 📐 Positional Encoding in Regimetry

This document explains how **positional encoding** is integrated into the `regimetry` project to support temporal awareness in windowed market data during embedding generation and regime clustering.

---

## 🔍 Why Positional Encoding?

In our regime detection pipeline, we generate rolling windows of shape `(num_windows, window_size, num_features)`. Many models — especially **Transformers** — are **position-agnostic**, meaning they require explicit encoding of the timestep within each window to model temporal dependencies.

Positional encoding injects information about each timestep's **absolute or relative position**, enabling effective modeling of sequential patterns.

---

## ✅ Supported Encoding Methods

### 1. **Sinusoidal (Fixed, Deterministic)**

- Based on the Transformer architecture from Vaswani et al. (2017)
- Uses alternating sine and cosine functions to represent position
- Encoding is deterministic and does not require training
- Compatible with both inference and training without parameterization

#### ✨ Styles:
- `"interleaved"`: `[sin₀, cos₀, sin₁, cos₁, ...]` (default)
- `"stacked"`: `[sin..., cos...]` (requires even `embedding_dim`)

```python
X_pe = PositionalEncoding.add(X, method='sinusoidal', encoding_style='interleaved')
```

> ✅ Uses `embedding_dim` from config if provided, otherwise infers from `X.shape[-1]`

---

### 2. **Learnable (Trainable Embedding)**

* Implements a `tf.keras.layers.Embedding` layer for `window_size` positions
* Each position maps to a learnable embedding vector
* Recommended for end-to-end trainable models, especially Transformers

```python
X_pe = PositionalEncoding.add(X, method='learnable')
```

> ✅ Uses `embedding_dim` from config (if provided) to project and encode; otherwise defaults to `X.shape[-1]`.  

---

## ⚙️ Dimension Resolution Logic

`embedding_dim` (passed internally as `learnable_dim`) is now **universally used** for both sinusoidal and learnable methods.

| Condition                              | Behavior                          |
| -------------------------------------- | --------------------------------- |
| `embedding_dim` defined in config.yaml | Used as override for encoding dim |
| `embedding_dim` not set                | Defaults to `X.shape[-1]`         |
| `stacked` sinusoidal + odd dim         | Pads input to even dimensionality |

This ensures consistent model behavior, especially when applying dimensionality projection before encoding.

---

## 🧮 Input Shape Expectations

All encodings expect input tensors of shape:

```python
(batch_size, window_size, embedding_dim)
```

This shape is enforced after any optional projection step and is used consistently throughout the embedding and clustering pipelines.

* `batch_size`: number of rolling windows
* `window_size`: temporal length of each window
* `embedding_dim`: feature size (either original or projected)

---

## 📦 Broadcasting Explained

Positional encoding shapes are always `(1, window_size, embedding_dim)` and broadcast across the batch axis:

```python
X_pe = X + tf.expand_dims(positional_encoding, axis=0)
```

This keeps memory usage efficient while maintaining shape alignment.

---

## 🧠 When to Use Each

| Scenario                              | Recommended Encoding |
| ------------------------------------- | -------------------- |
| Reproducible inference                | Sinusoidal           |
| End-to-end training with attention    | Learnable            |
| Small models or lightweight pipelines | Sinusoidal           |
| Transformer encoder block             | Learnable            |

---

## 📈 Visualization Tip

You can visualize the sinusoidal encoding patterns to see how position is encoded across dimensions.

### 🔍 Example

```python
import matplotlib.pyplot as plt
from regimetry.models.positional_encoding import PositionalEncoding

pe = PositionalEncoding.get_sinusoidal_encoding(seq_len=30, d_model=1273)
plt.imshow(pe[:, :128].numpy(), cmap='viridis', aspect='auto')
plt.colorbar()
plt.title("Sinusoidal Positional Encoding (First 128 Dimensions)")
plt.xlabel("Embedding Dimension")
plt.ylabel("Timestep")
```

> 🟢 This reveals the wave-like encoding structure across timesteps and shows how each dimension cycles differently.

---

## 🗂 Related Files

* `regimetry/models/positional_encoding.py`: core implementation
* `embedding_pipeline.py`: where positional encoding is applied
* `full_config.yaml`: where `encoding_method`, `encoding_style`, and `embedding_dim` are set
* `pad_to_even_dim()`: utility for stacked encoding support

---

## 🔁 Usage Reference

```python
# Automatically resolves embedding dim from config or input
X_pe = PositionalEncoding.add(
    X,
    method='sinusoidal',          # or 'learnable'
    encoding_style='stacked',     # optional for sinusoidal
    learnable_dim=config.embedding_dim  # used for both learnable and sinusoidal
)
```


