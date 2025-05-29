# 📘 Regime Clustering and Assignment Pipeline — Documentation

- [📘 Regime Clustering and Assignment Pipeline — Documentation](#-regime-clustering-and-assignment-pipeline--documentation)
  - [❓ Problem Statement](#-problem-statement)
  - [🤖 Why Spectral Clustering?](#-why-spectral-clustering)
  - [🔄 Cluster ID Alignment with the Hungarian Algorithm](#-cluster-id-alignment-with-the-hungarian-algorithm)
    - [🔍 Why Hungarian?](#-why-hungarian)
    - [⚙️ How It Works](#️-how-it-works)
  - [📁 Mapping Persistence Rules](#-mapping-persistence-rules)
    - [✅ Mapping is only written if:](#-mapping-is-only-written-if)
    - [❌ Mapping is **not** overwritten if:](#-mapping-is-not-overwritten-if)
  - [🔐 Key Design Decisions](#-key-design-decisions)
    - [1. **Baseline Assignments Must Be Stable**](#1-baseline-assignments-must-be-stable)
    - [2. **Mapping File Is Written Once**](#2-mapping-file-is-written-once)
    - [3. **Remapping Occurs Only if Needed**](#3-remapping-occurs-only-if-needed)
  - [🌊 Future: Handling Drift and New Clusters](#-future-handling-drift-and-new-clusters)


## ❓ Problem Statement

Unsupervised clustering algorithms like **Spectral Clustering** are inherently **non-deterministic**. This means that cluster labels (e.g., `Cluster_ID = 0, 1, 2...`) can be assigned differently even when the data or embedding space is nearly identical. Such inconsistencies pose major challenges when comparing regime behavior across runs, retraining cycles, or instruments. Without **stable cluster ID mappings**, metrics like transition matrices, cluster-based strategies, and interpretability become unreliable or misleading.

This pipeline addresses those issues by enforcing label stability through baseline alignment and safe remapping.

## 🤖 Why Spectral Clustering?

Despite its non-deterministic label assignment, **Spectral Clustering** was selected due to:

* Its ability to **capture non-convex structures** in embedding space
* Superior performance in separating **temporal and structural regimes**
* Compatibility with **precomputed affinity matrices** derived from high-dimensional transformer embeddings
* Excellent results in visual separability after t-SNE/UMAP reduction

> While clustering algorithms like **K-Means** and **DBSCAN** are popular and offer certain advantages — such as deterministic outputs or density-based grouping — they did not align as well with our objectives in this context. The regime structures we aim to uncover are often **non-convex, high-dimensional, and evolve over time**, characteristics that can challenge the assumptions these methods rely on.
>
> **Spectral Clustering**, by contrast, operates on a similarity graph and is better suited for capturing **complex relationships in embedded spaces**. In my experiments, it provided clearer separation of regimes when visualized with dimensionality reduction techniques, and aligned more closely with expected market behaviors.

> 📖 **Reference**:
> **[On Spectral Clustering: Analysis and an Algorithm](https://papers.nips.cc/paper/2092-on-spectral-clustering-analysis-and-an-algorithm.pdf)** — *Ng, Jordan, Weiss (2002)*
> *Key paper on spectral clustering. Uses graph Laplacians to identify non-convex clusters, suitable for latent regime discovery.*


## 🔄 Cluster ID Alignment with the Hungarian Algorithm

To fix the instability of Spectral Clustering labels, we introduced **cluster label alignment** using the **Hungarian algorithm**.


### 🔍 Why Hungarian?

The **Hungarian algorithm** (also known as the Kuhn–Munkres algorithm) is used to solve the **assignment problem** — it finds the minimum-cost matching between two sets. In our pipeline, it is used to **map new cluster labels to the closest corresponding baseline labels**, minimizing mismatch across runs.

> 📖 **Reference**:
> **[Hungarian Algorithm – Wikipedia](https://en.wikipedia.org/wiki/Hungarian_algorithm)**
> *Used to compute optimal one-to-one assignments between new and baseline cluster IDs.*


### ⚙️ How It Works

1. **Baseline Run**:

   * When no prior cluster metadata exists, the pipeline saves a **baseline cluster assignment**.
   * No alignment is applied, and no mapping file is created (labels are trusted as-is).

2. **Subsequent Runs**:

   * If a `baseline_cluster_assignments.csv` exists:

     * A cost matrix is computed between new and baseline cluster assignments.
     * The Hungarian algorithm maps each new cluster label to the most similar baseline label.
     * A mapping dictionary is generated (e.g., `{2 → 0, 0 → 1, 1 → 2}`).
     * This mapping is saved only if remapping occurred (i.e., labels are not already aligned).

3. **Mapping Output**:

   * If cluster ID remapping is needed, the pipeline writes:

     * **`regime_label_mapping.json`** — stores the aligned cluster ID mapping  
     * **`cluster_color_map.json`** — stores consistent visual colors for aligned clusters *(planned for future ??)*
  
---

## 📁 Mapping Persistence Rules

### ✅ Mapping is only written if:

* The Hungarian algorithm finds a non-identity mapping
* New clusters are introduced
* No prior mapping file exists

### ❌ Mapping is **not** overwritten if:

* The current run already matches the baseline
* There are no unmapped clusters
* The user did not explicitly authorize remapping

---

## 🔐 Key Design Decisions

### 1. **Baseline Assignments Must Be Stable**

* The baseline defines the “ground truth” mapping of cluster IDs for a given instrument and configuration.
* All future runs align to this to ensure consistent downstream analysis (e.g., Markov transitions, cluster roles).

### 2. **Mapping File Is Written Once**

* Prevents accidental drift due to reruns or different random seeds.
* Provides a traceable history of label evolution across experiments.
> 📂 Mapping files are saved to the configured baseline_metadata_dir, ensuring they persist even when --clean is used.


### 3. **Remapping Occurs Only if Needed**

* Reduces unnecessary computation and clutter.
* Ensures that aligned cluster IDs remain comparable across runs and reports.

---

## 🌊 Future: Handling Drift and New Clusters

* If a new run introduces **previously unseen clusters**, partial remapping may occur (needs testing).
* Such cases may signal **regime drift** or model evolution — these should be **flagged for manual or automated review**.
* Future extensions will include drift detection and version-controlled mapping evolution.
