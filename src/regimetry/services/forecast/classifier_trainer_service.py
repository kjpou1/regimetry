# regimetry/services/ForecastClassifierTrainerService.py

import os
from collections import Counter

import joblib
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

from regimetry.config import Config
from regimetry.logger_manager import LoggerManager

logging = LoggerManager.get_logger(__name__)


class ForecastClassifierTrainerService:
    def __init__(self, n_neighbors):
        self.config = Config()
        self.n_neighbors = n_neighbors

        self.model_output_path = self.config._resolve_path(
            os.path.join(self.config.output_dir, "knn_classifier.joblib")
        )

    def train(self, X, y):
        # === Sanity checks ===
        if len(X) != len(y):
            logging.error(
                f"❌ Feature-label mismatch: {len(X)} embeddings vs {len(y)} labels"
            )
            raise ValueError(
                "Length mismatch between input embeddings and cluster labels."
            )

        if np.isnan(y).any():
            num_nans = np.isnan(y).sum()
            logging.error(f"❌ Cluster labels contain {num_nans} NaN values.")
            raise ValueError(f"Cluster labels contain {num_nans} NaNs.")

        if np.isnan(X).any():
            logging.error("❌ Embeddings contain NaN values.")
            raise ValueError("Input embeddings contain NaNs.")

        logging.info(f"📦 Training KNN classifier on {len(X)} samples")
        cluster_counts = dict(sorted((int(k), v) for k, v in Counter(y).items()))
        logging.info(f"📊 Cluster distribution: {cluster_counts}")
        logging.info(f"🔧 KNN n_neighbors = {self.n_neighbors}")

        model = KNeighborsClassifier(n_neighbors=self.n_neighbors, weights="distance")
        model.fit(X, y)

        os.makedirs(os.path.dirname(self.model_output_path), exist_ok=True)
        joblib.dump(model, self.model_output_path)

        logging.info(f"✅ KNN model saved to: {self.model_output_path}")
        return model
