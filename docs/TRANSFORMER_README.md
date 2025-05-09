# Transformer Architecture in Regimetry (Draft)

This document outlines how Transformer-based models can be integrated into the `regimetry` framework for sequence modeling and regime classification.

---

## 📀 Input Format

All models expect inputs shaped as:

```python
(batch_size, window_size, num_features)
```

Where:

* `window_size`: Length of each rolling window (e.g. 30)
* `num_features`: Dimensionality of the feature space after transformation (e.g. 1273)

This aligns with the structure output from the `RollingWindowGenerator`.

---

## 🔁 Positional Encoding

Transformers are **position-agnostic**, so we must inject positional information.

Supported methods:

* **Sinusoidal**: Deterministic, non-trainable (suitable for static models)
* **Learnable**: Trainable embeddings (better for dynamic learning)

See [`POSITIONAL_ENCODING.md`](./POSITIONAL_ENCODING.md) for full documentation.

---

## 🛡️ Masking (Not Needed — For Future Support)

Transformers support **masking** to control attention behavior, but:

> ✅ **In `regimetry`, masking is not currently required** because all rolling windows are fixed-length and fully populated — no padding or causal decoding is needed.

We include this section for completeness and future extensibility.

### 1. Padding Mask (🔒 Optional)

Used to **ignore padded positions** in variable-length sequences (e.g., NLP):

```python
# Shape: (batch_size, seq_len)
mask = tf.cast(tf.math.not_equal(input_tensor, 0), tf.float32)
```

### 2. Look-Ahead Mask (🔒 Optional)

Used to **prevent attention to future timesteps** in autoregressive settings:

```python
look_ahead_mask = 1 - tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
```

These masks are not applied in our current implementation, but the code scaffolding exists should we expand to sequence prediction or token-based forecasting.

---

## 🧐 Model Block (Example)

```python
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, MultiHeadAttention, LayerNormalization, Dropout

def build_transformer_block(seq_len, d_model, num_heads, ff_dim):
    inputs = Input(shape=(seq_len, d_model))
    attn = MultiHeadAttention(num_heads=num_heads, key_dim=d_model)(inputs, inputs)
    attn = Dropout(0.1)(attn)
    out1 = LayerNormalization(epsilon=1e-6)(inputs + attn)

    ff = Dense(ff_dim, activation='relu')(out1)
    ff = Dense(d_model)(ff)
    ff = Dropout(0.1)(ff)
    out2 = LayerNormalization(epsilon=1e-6)(out1 + ff)

    return tf.keras.Model(inputs=inputs, outputs=out2)
```

---

## 🔌 Planned Integration

| Stage                   | Description                                 | Status        |
| ----------------------- | ------------------------------------------- | ------------- |
| Positional Encoding     | Injected before model input                 | ✅ Implemented |
| Transformer Encoder     | Lightweight encoder block                   | ➰ In Progress |
| Padding/Lookahead Masks | Optional support for future sequence models | 🕒 Not Yet    |
| Output Head             | Dense/MLP or classification/regime decoder  | ➰ In Progress |

---

## 🚧 Notes and Design Considerations

* You can experiment with multiple stacked encoder blocks
* Output can be pooled (e.g. mean or CLS token) before classification
* Causal decoding (e.g. for future prediction) would require masked attention and possibly sequence shifting


# Transformer Architecture in Regimetry

This document outlines how Transformer-based models can be integrated into the `regimetry` framework for sequence modeling and regime classification.

---

## 📐 Input Format

All models expect inputs shaped as:

```python
(batch_size, window_size, num_features)
```

Where:

* `window_size`: Length of each rolling window (e.g. 30)
* `num_features`: Dimensionality of the feature space after transformation (e.g. 1273)

This aligns with the structure output from the `RollingWindowGenerator`.

---

## 🔁 Positional Encoding

Transformers are **position-agnostic**, so we must inject positional information.

Supported methods:

* **Sinusoidal**: Deterministic, non-trainable (suitable for static models)
* **Learnable**: Trainable embeddings (better for dynamic learning)

See [`POSitional_ENCODING.md`](./POSITIONAL_ENCODING.md) for full documentation.

---

## 🛡️ Masking (Not Needed — For Future Support)

Transformers support **masking** to control attention behavior, but:

> ✅ **In `regimetry`, masking is not currently required** because all rolling windows are fixed-length and fully populated — no padding or causal decoding is needed.

We include this section for completeness and future extensibility.

### 1. Padding Mask (🔒 Optional)

Used to **ignore padded positions** in variable-length sequences (e.g., NLP):

```python
# Shape: (batch_size, seq_len)
mask = tf.cast(tf.math.not_equal(input_tensor, 0), tf.float32)
```

### 2. Look-Ahead Mask (🔒 Optional)

Used to **prevent attention to future timesteps** in autoregressive settings:

```python
look_ahead_mask = 1 - tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
```

These masks are not applied in our current implementation, but the code scaffolding exists should we expand to sequence prediction or token-based forecasting.

---

## 🧠 Model Block (Example)

```python
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, MultiHeadAttention, LayerNormalization, Dropout

def build_transformer_block(seq_len, d_model, num_heads, ff_dim):
    inputs = Input(shape=(seq_len, d_model))
    attn = MultiHeadAttention(num_heads=num_heads, key_dim=d_model)(inputs, inputs)
    attn = Dropout(0.1)(attn)
    out1 = LayerNormalization(epsilon=1e-6)(inputs + attn)

    ff = Dense(ff_dim, activation='relu')(out1)
    ff = Dense(d_model)(ff)
    ff = Dropout(0.1)(ff)
    out2 = LayerNormalization(epsilon=1e-6)(out1 + ff)

    return tf.keras.Model(inputs=inputs, outputs=out2)
```

---

## 🔌 Planned Integration

| Stage                   | Description                                 | Status         |
| ----------------------- | ------------------------------------------- | -------------- |
| Positional Encoding     | Injected before model input                 | ✅ Implemented  |
| Transformer Encoder     | Lightweight encoder block                   | 🔄 In Progress |
| Padding/Lookahead Masks | Optional support for future sequence models | 🕒 Not Yet     |
| Output Head             | Dense/MLP or classification/regime decoder  | 🔄 In Progress |

---

## 🚧 Notes and Design Considerations

* You can experiment with multiple stacked encoder blocks
* Output can be pooled (e.g. mean or CLS token) before classification
* Causal decoding (e.g. for future prediction) would require masked attention and possibly sequence shifting


