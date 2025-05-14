
# 📊 Transformer Encoder in Regimetry vs. LLM Architectures

This document compares the lightweight Transformer encoder used in the `regimetry` framework with traditional large language models (LLMs) such as **BERT** and **GPT**.

---

## 🧠 Purpose & Use Case

| Feature       | `regimetry` Transformer Encoder             | LLMs (e.g., BERT, GPT)                               |
| ------------- | ------------------------------------------- | ---------------------------------------------------- |
| **Goal**      | Extract latent *market regime embeddings*   | Generate or classify *natural language*              |
| **Usage**     | Downstream clustering & regime segmentation | Text generation, classification, question answering  |
| **Task Type** | Unsupervised time-series representation     | Supervised pretraining (BERT) / autoregressive (GPT) |

---

## 🧮 Input Format

| Feature         | `regimetry`                               | LLMs                                 |
| --------------- | ----------------------------------------- | ------------------------------------ |
| Input           | `(batch_size, window_size, num_features)` | `(batch_size, seq_len)` (token IDs)  |
| Token Type      | Continuous multivariate features          | Discrete tokens (integers)           |
| Sequence Length | Fixed window size (e.g., 30)              | Typically fixed (e.g., 512 for BERT) |

---

## 🔁 Positional Encoding

| Type       | `regimetry`            | LLMs                             |
| ---------- | ---------------------- | -------------------------------- |
| Sinusoidal | ✅ Supported (default)  | ✅ Common in original Transformer |
| Learnable  | ✅ Supported (optional) | ✅ Standard in BERT/GPT variants  |

---

## 🧩 Encoder Architecture

| Layer         | `regimetry` Transformer       | LLMs                                    |
| ------------- | ----------------------------- | --------------------------------------- |
| Attention     | MultiHead Self-Attention      | MultiHead Self-Attention                |
| Normalization | Post-attention & post-FFN     | Same                                    |
| Feedforward   | Conv1D (1×1) FFN blocks       | Dense FFN (linear layers)               |
| Pooling       | ✅ `GlobalAveragePooling1D()`  | ❌ BERT uses \[CLS]; GPT uses last token |
| Output        | Fixed-length embedding vector | Token-level or sequence output          |

---

## 🛡️ Masking Strategy

| Mask Type       | `regimetry`                 | LLMs                           |
| --------------- | --------------------------- | ------------------------------ |
| Padding Mask    | ❌ Not needed (no padding)   | ✅ BERT                         |
| Look-Ahead Mask | ❌ Not used (not generative) | ✅ GPT                          |
| Rationale       | Fixed-length time windows   | Variable-length, causal models |

---

## 🎯 Output Head

| Output Head | `regimetry`                | LLMs                                 |
| ----------- | -------------------------- | ------------------------------------ |
| Purpose     | Embedding for clustering   | Classification (BERT) / logits (GPT) |
| Example     | 256-dim vector via pooling | Vocabulary logits or class labels    |

> 🧠 *`regimetry` intentionally excludes a supervised output head to focus on **representation learning**, not prediction.*

---

## 📏 Model Size & Complexity

| Dimension            | `regimetry` Encoder          | LLMs (e.g., BERT-base, GPT-2) |
| -------------------- | ---------------------------- | ----------------------------- |
| # Transformer Blocks | 2–4                          | 12–96                         |
| Parameters           | \~0.5M–2M                    | 110M (BERT) → 175B+ (GPT-3)   |
| Training Data        | None (unsupervised use only) | Massive text corpora          |
| Fine-tuning          | ❌ Not required               | ✅ Often required per task     |

---

## ✅ Summary Table

| Feature             | `regimetry` Transformer         | BERT / GPT (LLMs)                |
| ------------------- | ------------------------------- | -------------------------------- |
| Input type          | Continuous time-series          | Discrete text tokens             |
| Attention type      | Self-attention (bidirectional)  | BERT: bidirectional, GPT: causal |
| Output              | Vector embedding (unsupervised) | Class logits / tokens            |
| Masking             | None                            | Padding / causal masks           |
| Positional encoding | Sinusoidal or learnable         | Usually learnable                |
| Pooling             | Global average                  | \[CLS] or final token            |
| Output head         | ❌ None                          | ✅ Present                        |
| Size & depth        | Shallow (2–4 layers)            | Deep (12–96+ layers)             |

---

## 🧭 Design Philosophy

**`regimetry`** uses Transformer blocks not for language modeling but for **stateful regime representation**.

* Lightweight and interpretable
* Optimized for unsupervised embedding quality
* Avoids complexity of generative decoding or task-specific heads
* Outputs are clustered post hoc (e.g., Spectral Clustering)
