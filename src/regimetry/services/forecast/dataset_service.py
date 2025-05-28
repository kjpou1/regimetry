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

    def build_dataset(self) -> ForecastDataset:
        """
        Loads embeddings, validates metadata, loads cluster labels,
        and builds the rolling window forecast dataset.
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

        # Resolve window and stride with CLI override warnings
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
        assert len(cluster_labels) == embeddings.shape[0], (
            f"❌ Length mismatch: {len(cluster_labels)} cluster labels vs "
            f"{embeddings.shape[0]} embeddings."
        )
        logging.info("✅ Cluster assignment checks passed.")

        # ✅ Build dataset using refactored logic
        X, Y, Yc = build_embedding_forecast_dataset(
            embeddings, cluster_labels, self.window_size, self.stride
        )

        logging.info(
            f"✅ Built forecast dataset: X={X.shape}, Y={Y.shape}, Yc={Yc.shape}"
        )

        return ForecastDataset(
            X=X,
            Y=Y,
            Y_cluster=Yc,
            embeddings=embeddings,
            cluster_labels=cluster_labels,
            metadata=metadata,
            window_size=self.window_size,
            stride=self.stride,
        )
