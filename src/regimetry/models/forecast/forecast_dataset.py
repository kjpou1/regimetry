# src/regimetry/models/forecast/forecast_dataset.py

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class ForecastDataset:
    """
    Represents a rolling window training set for forecasting E[t+1] and predicting Cluster_ID[t+1].

    Attributes:
        X (np.ndarray): Input windows of shape (N, window_size, embedding_dim)
        Y (np.ndarray): Next-step embeddings of shape (N, embedding_dim)
        Y_cluster (np.ndarray): Next-step cluster IDs of shape (N,)
        embeddings (np.ndarray): Full normalized embedding matrix (T, embedding_dim)
        cluster_labels (np.ndarray): Original Cluster_ID[t] labels (T,)
        metadata (dict): Metadata dictionary loaded from embedding_metadata.json
        window_size (int): Window size used to build the dataset
        stride (int): Stride used for rolling window generation
    """

    X: np.ndarray
    Y: np.ndarray
    Y_cluster: np.ndarray
    embeddings: np.ndarray
    cluster_labels: np.ndarray
    metadata: dict
    window_size: int
    stride: int

    def summary(self) -> str:
        return (
            f"🧾 ForecastDataset summary:\n"
            f"  → X shape:         {self.X.shape}\n"
            f"  → Y shape:         {self.Y.shape}\n"
            f"  → Y_cluster shape: {self.Y_cluster.shape}\n"
            f"  → Full embeddings: {self.embeddings.shape}\n"
            f"  → Cluster labels:  {self.cluster_labels.shape}\n"
            f"  → Window/Stride:   {self.window_size} / {self.stride}\n"
        )

    def to_dataframe(self, reduce_x: str = "mean") -> pd.DataFrame:
        """
        Converts the dataset into a tabular DataFrame for inspection.
        Reduces X windows to summary features (mean or last step).
        """
        if reduce_x == "mean":
            X_flat = self.X.mean(axis=1)
        elif reduce_x == "last":
            X_flat = self.X[:, -1, :]
        else:
            raise ValueError("reduce_x must be 'mean' or 'last'")

        df = pd.DataFrame(X_flat)
        df.columns = [f"x_{i}" for i in range(df.shape[1])]
        df["cluster_id"] = self.Y_cluster
        return df
