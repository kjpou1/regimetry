import numpy as np


def build_embedding_forecast_dataset(
    embeddings: np.ndarray, cluster_labels: np.ndarray, window_size: int, stride: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Constructs a supervised dataset from a sequence of embedding vectors and cluster labels.

    Exactly — the pipeline is doing supervised dataset construction where:

    * `X`: a rolling window of past embeddings → `[E_{t−W+1}, ..., E_t]`
    * `Y`: the immediate *next* embedding vector → `E_{t+1}`
    * `Yc`: the associated *next* cluster label → `Cluster_ID[t+1]`

    This builds a supervised learning problem:

        “Given the last W embeddings, predict what the next embedding and regime will be.”

    This supports two downstream tasks:

    1. **Embedding Forecasting** (`X → Y`)
    2. **Regime Classification** (`Y → Cluster_ID[t+1]`)

    Example:
        ```python
        E_hat = model.predict(X_latest_window)
        predicted_cluster = knn.predict([E_hat])[0]
        ```

    Returns:
        Tuple of (X, Y, Yc) numpy arrays
    """
    W, S = window_size, stride
    X, Y, Y_cluster = [], [], []

    # Iterate over the embedding timeline using a rolling window
    # Start at index W-1 so the first full window is valid (0 to W-1)
    # Stop at len-1 so we can still access t+1 for the forecast target
    for t in range(W - 1, len(embeddings) - 1, S):
        # Skip if the cluster label for t+1 is missing
        if np.isnan(cluster_labels[t + 1]):
            continue

        # Extract the window of embeddings ending at time t
        x_window = embeddings[t - W + 1 : t + 1]

        # Ensure the window has exactly W elements (robustness guard)
        if x_window.shape[0] != W:
            continue

        # Append input window
        X.append(x_window)

        # Append next-step embedding (E[t+1])
        Y.append(embeddings[t + 1])

        # Append next-step cluster label (Cluster_ID[t+1])
        Y_cluster.append(cluster_labels[t + 1])

    # Convert all lists to NumPy arrays for model compatibility
    X = np.array(X)
    Y = np.array(Y)
    Y_cluster = np.array(Y_cluster)

    # Raise an error if no valid samples were generated
    if X.shape[0] == 0:
        raise ValueError("❌ No valid training samples generated.")

    return X, Y, Y_cluster
