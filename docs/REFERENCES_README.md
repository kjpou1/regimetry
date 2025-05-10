# 📚 Regimetry Reading List: Transformers + Unsupervised Regime Detection

This guide outlines the foundational literature and tools used in the `regimetry` pipeline — from Transformer-based time-series modeling to unsupervised clustering via spectral methods.

---

## 🔁 Transformer Encoders for Sequences

- **[Attention is All You Need](https://arxiv.org/abs/1706.03762)** — Vaswani et al., 2017  
  *The original paper introducing Transformers. Core concepts: self-attention, multi-head attention, and positional encoding.*

- **[A Transformer-based Framework for Multivariate Time Series Representation Learning](https://arxiv.org/abs/2010.02803)** — Zerveas et al., 2021  
  *Adapts Transformers for unsupervised representation learning on multivariate time-series windows. Uses mean pooling (no CLS token). Very similar to `regimetry`'s approach.*  
  [GitHub Code](https://github.com/gzerveas/mvts_transformer)

- **[Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting](https://arxiv.org/abs/2012.07436)** — Wu et al., 2021  
  *Scalable Transformer for long-term sequence prediction. Introduces ProbSparse self-attention for efficiency.*

---

## 🧠 Unsupervised Embedding & Pooling Strategies

- **[A Structured Self-Attentive Sentence Embedding](https://arxiv.org/abs/1703.03130)** — Lin et al., 2017  
  *Demonstrates pooling mechanisms (mean, attention) over sequence representations. Justifies using mean pooling for your embeddings.*

- **[SimCLR: A Simple Framework for Contrastive Learning](https://arxiv.org/abs/2002.05709)** — Chen et al., 2020  
  *Contrastive representation learning using data augmentations and projection heads. Optional if `regimetry` evolves toward contrastive learning.*

---

## 📈 Embedding Clustering & Visualization

- **[On Spectral Clustering: Analysis and an Algorithm](https://papers.nips.cc/paper/2092-on-spectral-clustering-analysis-and-an-algorithm.pdf)** — Ng, Jordan, Weiss, 2002  
  *Key paper on spectral clustering. Uses graph Laplacians to identify non-convex clusters, suitable for latent regime discovery.*

- **[DeepCluster: Unsupervised Learning of Visual Features by Clustering](https://arxiv.org/abs/1807.05520)** — Caron et al., 2018  
  *Combines clustering and representation learning in a feedback loop. Useful for future extensions of `regimetry`.*

- **[Visualizing Data using t-SNE](https://www.jmlr.org/papers/volume9/vandermaaten08a/vandermaaten08a.pdf)** — van der Maaten & Hinton, 2008  
  *Dimensionality reduction technique widely used for visualizing high-dimensional embeddings (e.g. regime clusters).*

---

## 🧰 Tools & Examples

- **[tsai: Time-Series AI Toolkit](https://github.com/timeseriesAI/tsai)**  
  *FastAI-style PyTorch toolkit for time-series models. Includes Transformer, InceptionTime, and ResNet variants.*

- **[Keras Time Series Transformer Example](https://keras.io/examples/timeseries/timeseries_classification_transformer/)**  
  *Official Keras example demonstrating how to classify time-series windows using a Transformer encoder.  
  Note: This example does **not** include positional encoding — a design choice acknowledged in [GitHub Issue #1894](https://github.com/keras-team/keras-io/issues/1894).*

- **[TensorFlow Transformer Tutorial](https://www.tensorflow.org/text/tutorials/transformer)** — TensorFlow.org  
  *Step-by-step implementation of a Transformer model using TensorFlow/Keras. Includes sinusoidal positional encoding using a stacked sine/cosine format (concatenated rather than interleaved), multi-head attention, encoder blocks, and end-to-end training.*
  
- **[scikit-learn Spectral Clustering Docs](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.SpectralClustering.html)**  
  *API documentation for Spectral Clustering used in `regimetry`.*

---
