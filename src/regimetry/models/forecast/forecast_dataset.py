from dataclasses import dataclass
from typing import Optional

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
        X_val (Optional[np.ndarray]): Optional validation input windows
        Y_val (Optional[np.ndarray]): Optional validation next-step embeddings
        Y_cluster_val (Optional[np.ndarray]): Optional validation cluster targets
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
    X_val: Optional[np.ndarray] = None
    Y_val: Optional[np.ndarray] = None
    Y_cluster_val: Optional[np.ndarray] = None

    def summary(self) -> str:
        lines = [
            "🧾 ForecastDataset summary:",
            f"  → X shape:         {self.X.shape}",
            f"  → Y shape:         {self.Y.shape}",
            f"  → Y_cluster shape: {self.Y_cluster.shape}",
            f"  → Full embeddings: {self.embeddings.shape}",
            f"  → Cluster labels:  {self.cluster_labels.shape}",
            f"  → Window/Stride:   {self.window_size} / {self.stride}",
            f"  → Unique Clusters: {np.unique(self.Y_cluster)}",
        ]
        if self.X_val is not None:
            val_clusters = np.unique(self.Y_cluster_val)
            train_clusters = np.unique(self.Y_cluster)
            unseen = sorted(int(c) for c in set(val_clusters) - set(train_clusters))

            lines.extend(
                [
                    f"  → X_val shape:     {self.X_val.shape}",
                    f"  → Y_val shape:     {self.Y_val.shape}",
                    f"  → Y_cluster_val shape:    {self.Y_cluster_val.shape}",
                    f"  → Unique Clusters (val): {val_clusters}",
                ]
            )
            if unseen:
                lines.append(f"  ⚠️ Val-only clusters:    {unseen}")

        return "\n".join(lines)

    def to_dataframe(
        self, reduce_x: str = "mean", use_validation: bool = False
    ) -> pd.DataFrame:
        """
        Converts the dataset into a tabular DataFrame for inspection.
        Reduces X windows to summary features (mean or last step).

        Args:
            reduce_x (str): 'mean' or 'last' to reduce the window.
            use_validation (bool): Use validation set instead of train.
        """
        X_data = self.X_val if use_validation and self.X_val is not None else self.X
        Yc_data = (
            self.Y_cluster_val
            if use_validation and self.Y_cluster_val is not None
            else self.Y_cluster
        )

        if reduce_x == "mean":
            X_flat = X_data.mean(axis=1)
        elif reduce_x == "last":
            X_flat = X_data[:, -1, :]
        else:
            raise ValueError("reduce_x must be 'mean' or 'last'")

        df = pd.DataFrame(X_flat)
        df.columns = [f"x_{i}" for i in range(df.shape[1])]
        df["cluster_id"] = Yc_data
        return df
