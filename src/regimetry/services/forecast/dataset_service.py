import json
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize

from regimetry.config.config import Config
from regimetry.logger_manager import LoggerManager
from regimetry.models.forecast.forecast_dataset import ForecastDataset
from regimetry.utils.forcast_utils import build_embedding_forecast_dataset

logging = LoggerManager.get_logger(__name__)


class ForecastDatasetService:
    """
    Builds the supervised dataset for forecasting E[t+1] and predicting Cluster_ID[t+1]
    using normalized embeddings and cluster assignments.
    """

    def __init__(self):
        self.config = Config()
        self.embedding_file = self.config.embedding_file
        self.metadata_file = self.config.embedding_metadata_path
        self.cluster_assignment_path = self.config.cluster_assignment_path
        self.window_size = self.config.window_size
        self.stride = self.config.stride

    def build_dataset(self, validation_split: float = 0.0) -> ForecastDataset:
        """
        Loads embeddings, validates metadata, loads cluster labels,
        and builds the rolling window forecast dataset with optional time-based validation split.
        """
        logging.info("📥 Loading and validating embeddings/metadata...")

        if not os.path.exists(self.embedding_file):
            raise FileNotFoundError(
                f"❌ embeddings.npy not found: {self.embedding_file}"
            )
        if not os.path.exists(self.metadata_file):
            raise FileNotFoundError(f"❌ metadata file not found: {self.metadata_file}")
        if not os.path.exists(self.cluster_assignment_path):
            raise FileNotFoundError(
                f"❌ cluster_assignment.csv not found: {self.cluster_assignment_path}"
            )

        embeddings = np.load(self.embedding_file)
        embeddings = normalize(embeddings, norm="l2", axis=1)

        with open(self.metadata_file, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        expected_shape = (metadata["n_samples"], metadata["embedding_dim"])
        if embeddings.shape != expected_shape:
            raise ValueError(
                f"❌ Embedding shape mismatch: expected {expected_shape}, found {embeddings.shape}"
            )

        # Resolve window and stride
        cli_window = self.config.window_size
        cli_stride = self.config.stride
        meta_window = metadata.get("window_size")
        meta_stride = metadata.get("stride")

        self.window_size = cli_window or meta_window
        self.stride = cli_stride or meta_stride

        if self.window_size is None:
            raise ValueError("❌ window_size must be specified via CLI or metadata.")
        if self.stride is None:
            raise ValueError("❌ stride must be specified via CLI or metadata.")

        if cli_window and meta_window and cli_window != meta_window:
            logging.warning(
                f"⚠️ CLI window_size ({cli_window}) overrides metadata value ({meta_window})"
            )
        if cli_stride and meta_stride and cli_stride != meta_stride:
            logging.warning(
                f"⚠️ CLI stride ({cli_stride}) overrides metadata value ({meta_stride})"
            )

        cluster_df = pd.read_csv(self.cluster_assignment_path, encoding="utf-8")
        cluster_labels = cluster_df["Cluster_ID"].dropna().values

        num_nans = np.isnan(cluster_labels).sum()
        assert num_nans == 0, f"❌ Cluster labels contain {num_nans} NaN values."
        assert (
            len(cluster_labels) == embeddings.shape[0]
        ), f"❌ Length mismatch: {len(cluster_labels)} cluster labels vs {embeddings.shape[0]} embeddings."
        logging.info("✅ Cluster assignment checks passed.")

        # ✅ Build dataset
        X, Y, Y_cluster = build_embedding_forecast_dataset(
            embeddings, cluster_labels, self.window_size, self.stride
        )
        logging.info(
            f"✅ Built forecast dataset: X={X.shape}, Y={Y.shape}, Yc={Y_cluster.shape}"
        )

        # ✂️ Time-based validation split
        if not (0.0 <= validation_split < 1.0):
            raise ValueError(
                f"❌ validation_split must be between 0.0 and 1.0 (exclusive of 1.0), got {validation_split}"
            )

        if validation_split > 0.0:
            split_idx = int(len(X) * (1 - validation_split))
            X_train, X_val = X[:split_idx], X[split_idx:]
            Y_train, Y_val = Y[:split_idx], Y[split_idx:]
            Y_cluster_train, Y_cluster_val = (
                Y_cluster[:split_idx],
                Y_cluster[split_idx:],
            )
        else:
            X_train, X_val = X, None
            Y_train, Y_val = Y, None
            Y_cluster_train, Y_cluster_val = Y_cluster, None

        return ForecastDataset(
            X=X_train,
            Y=Y_train,
            Y_cluster=Y_cluster_train,
            X_val=X_val,
            Y_val=Y_val,
            Y_cluster_val=Y_cluster_val,
            embeddings=embeddings,
            cluster_labels=cluster_labels,
            metadata=metadata,
            window_size=self.window_size,
            stride=self.stride,
        )
