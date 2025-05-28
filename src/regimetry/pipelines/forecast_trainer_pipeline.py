import json
import os
from datetime import datetime
from os import path

import joblib
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import normalize
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

from regimetry.config.config import Config
from regimetry.logger_manager import LoggerManager
from regimetry.services.forecast.dataset_service import ForecastDatasetService

logging = LoggerManager.get_logger(__name__)


class ForecastTrainerPipeline:
    """
    ForecastTrainerPipeline

    This pipeline trains a supervised model to predict the next embedding vector
    (`E[t+1]`) from a rolling window of prior embeddings. It also trains a KNN classifier
    to map embeddings to cluster IDs (`Cluster_ID[t]`) for real-time regime forecasting.
    """

    def __init__(self):
        self.config = Config()

        self.dataset_service = ForecastDatasetService()
        # # Resolve mandatory paths
        # if not path.exists(self.config.embedding_file):
        #     raise FileNotFoundError(
        #         f"❌ embeddings.npy not found: {self.config.embedding_file}"
        #     )
        # if not path.exists(self.config.embedding_metadata_path):
        #     raise FileNotFoundError(
        #         f"❌ metadata file not found: {self.config.embedding_metadata_path}"
        #     )
        # if not path.exists(self.config.cluster_assignment_path):
        #     raise FileNotFoundError(
        #         f"❌ cluster_assignment.csv not found: {self.config.cluster_assignment_path}"
        #     )

        # Output location
        self.output_dir = self.config.output_dir or os.path.join(
            self.config.BASE_DIR, "forecast_models", self.config.instrument
        )
        os.makedirs(self.output_dir, exist_ok=True)

    def run(self):
        logging.info("🚀 Starting forecast training pipeline")

        # STEP 1–2: Load embeddings and build training dataset
        dataset = self.dataset_service.build_dataset()

        X = dataset.X
        Y = dataset.Y
        Yc = dataset.Y_cluster
        embeddings = dataset.embeddings
        cluster_labels = dataset.cluster_labels

        logging.info(dataset.summary())

        # # STEP 3: Train forecaster model
        # model = get_model(
        #     self.config.model_type,
        #     input_shape=(self.config.window_size, embeddings.shape[1]),
        #     output_dim=embeddings.shape[1],
        # )
        # model.compile(optimizer=Adam(), loss="cosine_similarity")

        # logging.info(f"🧠 Training model: {self.config.model_type}")
        # model.fit(
        #     X,
        #     Y,
        #     epochs=300,
        #     batch_size=64,
        #     callbacks=[ReduceLROnPlateau(patience=10)],
        #     verbose=1,
        # )

        # # STEP 4: Train KNN classifier on original embeddings
        # valid_mask = ~np.isnan(cluster_labels)
        # knn = KNeighborsClassifier(
        #     n_neighbors=self.config.n_neighbors, weights="distance"
        # )
        # knn.fit(embeddings[valid_mask], cluster_labels[valid_mask])

        # # STEP 5: Save all artifacts
        # model_path = os.path.join(self.output_dir, "embedding_forecaster.h5")
        # knn_path = os.path.join(self.output_dir, "knn_cluster_classifier.pkl")
        # summary_path = os.path.join(self.output_dir, "training_summary.json")

        # model.save(model_path)
        # joblib.dump(knn, knn_path)

        # summary = {
        #     "instrument": self.config.instrument,
        #     "model_type": self.config.model_type,
        #     "embedding_dim": embeddings.shape[1],
        #     "window_size": self.config.window_size,
        #     "stride": self.config.stride,
        #     "n_samples_used": X.shape[0],
        #     "n_clusters": int(np.nanmax(cluster_labels) + 1),
        #     "timestamp": datetime.utcnow().isoformat(),
        # }

        # with open(summary_path, "w", encoding="utf-8") as f:
        #     json.dump(summary, f, indent=2)

        # logging.info(f"💾 Forecaster model saved: {model_path}")
        # logging.info(f"💾 KNN model saved:        {knn_path}")
        # logging.info(f"📄 Training summary saved: {summary_path}")
        # logging.info("✅ Forecast training pipeline completed.")
