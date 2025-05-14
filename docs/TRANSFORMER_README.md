
# Transformer Architecture in `regimetry`

This document outlines how Transformer-based models are integrated into the `regimetry` framework for unsupervised sequence modeling and regime clustering.

---

## 📐 Input Format

All models expect inputs shaped as:

```python
(batch_size, window_size, num_features)
```

Where:

* `window_size`: Length of each rolling window (e.g. 30)
* `num_features`: Dimensionality of the feature space after transformation (e.g. 1273)

These inputs are produced by the rolling window pipeline and directly passed to the embedding model.

---

## 🔁 Positional Encoding

Transformers are **position-agnostic**, so we inject positional information.

### Supported Methods:

* **Sinusoidal**: Deterministic, non-trainable (default)
* **Learnable**: Trainable embeddings (optional via config)

See [`POSITIONAL_ENCODING.md`](./POSITIONAL_ENCODING.md) for full details.

---

## 🛡️ Masking (Not Required)

Transformers support **masking** to control attention behavior, but:

> ✅ In `regimetry`, masking is not currently required.

All windows are fixed-length and fully populated. We do not pad or perform causal decoding.

### Optional Mask Types (Not Used Yet):

**1. Padding Mask**

```python
mask = tf.cast(tf.math.not_equal(input_tensor, 0), tf.float32)
```

**2. Look-Ahead Mask**

```python
look_ahead_mask = 1 - tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
```

These are reserved for future support in autoregressive models.

---

## 🧠 Current Model Architecture

The core embedding model is a stack of Transformer encoder blocks, ending in a global pooling layer.

```python
def _build_model(self):
    """
    Build and return a Transformer encoder model for embedding extraction.
    """
    head_size = self.config.get("head_size", 256)
    num_heads = self.config.get("num_heads", 4)
    ff_dim = self.config.get("ff_dim", 128)
    num_blocks = self.config.get("num_transformer_blocks", 2)
    dropout = self.config.get("dropout", 0.1)

    inputs = keras.Input(shape=self.input_shape)
    x = inputs
    for _ in range(num_blocks):
        x = self._transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

    x = layers.GlobalAveragePooling1D()(x)
    return keras.Model(inputs=inputs, outputs=x, name="UnsupervisedTransformerEncoder")
```

Each encoder block contains attention, residuals, and pointwise feedforward layers:

```python
def _transformer_encoder(self, inputs, head_size, num_heads, ff_dim, dropout):
    x = layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(inputs, inputs)
    x = layers.Dropout(dropout)(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    res = x + inputs

    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(res)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    return x + res
```

---

## 🔌 Integration Status

| Stage                   | Description                                 | Status        |
| ----------------------- | ------------------------------------------- | ------------- |
| **Positional Encoding** | Injected before Transformer input           | ✅ Implemented |
| **Transformer Encoder** | Multi-block attention + conv FF layers      | ✅ Implemented |
| **Masking Support**     | Padding / look-ahead masks (optional)       | 🕒 Not Needed |
| **Output Head**         | GlobalAvgPooling → used as embedding vector | ✅ Implemented |

---

## 🧭 Design Notes

* The current model is **unsupervised** — output vectors are clustered via Spectral Clustering.
* Global average pooling is used to compress temporal signals into fixed-length regime embeddings.
* The encoder is **modular** — you can easily swap in additional attention heads, deeper stacks, or alternative pooling strategies.
* Future versions may introduce decoder blocks or contrastive learning heads.
