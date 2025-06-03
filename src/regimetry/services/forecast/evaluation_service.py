import json
import os
from collections import Counter
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from tensorflow.keras.models import load_model

from regimetry.config.config import Config
from regimetry.logger_manager import LoggerManager
from regimetry.models.forecast.forecast_dataset import ForecastDataset

logging = LoggerManager.get_logger(__name__)


class ForecastEvaluationService:
    """
    ForecastEvaluationService

    This service handles the evaluation of a trained forecast model and KNN classifier
    used for regime prediction. It computes prediction accuracy, embedding loss (MSE),
    per-cluster metrics, and summary statistics for the latest forecast.

    Key Responsibilities:
        - Load trained forecast model (with fallback to non-best version)
        - Load trained KNN classifier for cluster mapping
        - Load training history for loss tracking
        - Evaluate full sequence prediction performance
        - Predict the next embedding and cluster
        - Provide metrics and summaries for downstream logging or reporting

    Args:
        forecast_model_path (str): Path to the trained forecast model (final).
        best_model_path (str): Path to the best model checkpoint (if exists).
        classifier_model_path (str): Path to the trained KNN classifier.
        history_path (str): Path to training history JSON (loss curves).
        dataset (ForecastDataset): ForecastDataset object containing input/target embeddings and cluster labels.
    """

    def __init__(
        self,
        forecast_model_path: str,
        best_model_path: str,
        classifier_model_path: str,
        history_path: str,
        dataset: ForecastDataset,
    ):
        self.forecast_model_path = forecast_model_path
        self.best_model_path = best_model_path
        self.classifier_model_path = classifier_model_path
        self.history_path = history_path
        self.forecast_model = None
        self.knn_model = None
        self.dataset = dataset
        self.config = Config()

        self._load_models()
        self._load_history()

    def _load_models(self):
        """Loads the forecast model (prefers best) and the trained KNN classifier."""
        if os.path.exists(self.best_model_path):
            self.forecast_model = load_model(self.best_model_path)
            logging.info(f"✅ Loaded best forecast model: {self.best_model_path}")
        elif os.path.exists(self.forecast_model_path):
            self.forecast_model = load_model(self.forecast_model_path)
            logging.warning(
                f"⚠️ Best model not found. Loaded final model: {self.forecast_model_path}"
            )
        else:
            raise FileNotFoundError(
                "No forecast model found (best or final). Please train a model first."
            )

        self.knn_model = joblib.load(self.classifier_model_path)
        logging.info(f"✅ Loaded KNN classifier: {self.classifier_model_path}")

    def _load_history(self):
        """Loads training history JSON and extracts key loss metrics."""
        if not os.path.exists(self.history_path):
            raise FileNotFoundError(f"❌ History file not found: {self.history_path}")

        with open(self.history_path, "r", encoding="utf-8") as f:
            self.history = json.load(f)

        if "loss" not in self.history or not self.history["loss"]:
            raise ValueError("❌ History file is missing 'loss' values.")

        self.epochs_run = len(self.history["loss"])
        self.final_train_loss = self.history["loss"][-1]
        self.final_val_loss = self.history.get("val_loss", [None])[-1]

        logging.info(
            f"📈 History loaded: {self.epochs_run} epochs, final loss = {self.final_train_loss:.6f}"
            + (f", val_loss = {self.final_val_loss:.6f}" if self.final_val_loss else "")
        )

    def evaluate(self):
        """
        Runs full evaluation:
            - Predicts next embedding and maps to cluster
            - Computes embedding MSE for sequence prediction
            - Computes classification accuracy
            - Generates confusion matrix and per-cluster metrics
        """
        logging.info("🔮 Predicting next embedding Ê[t+1]...")
        embeddings = self.dataset.embeddings
        window_size = self.dataset.window_size

        recent_sequence = embeddings[-window_size:]
        X_input = np.expand_dims(recent_sequence, axis=0)
        E_hat = self.forecast_model.predict(X_input, verbose=0)[0]

        logging.info("🧠 Predicting cluster ID from Ê[t+1]...")
        predicted_prob = self.knn_model.predict_proba([E_hat])
        predicted_cluster = self.knn_model.predict([E_hat])[0]

        # Evaluate prediction quality on entire dataset
        logging.info("📊 Running full sequence evaluation...")
        X = self.dataset.X
        Y_true = self.dataset.Y
        Y_cluster_true = self.dataset.Y_cluster

        Y_pred = self.forecast_model.predict(X, verbose=0)
        self.embedding_mse = float(np.mean(np.square(Y_pred - Y_true)))

        self.predicted_clusters = self.knn_model.predict(Y_pred)
        self.true_clusters = Y_cluster_true
        self.accuracy = float(
            accuracy_score(self.true_clusters, self.predicted_clusters)
        )

        # Determine the full set of expected cluster labels (based on config)
        all_labels = sorted(set(np.unique(self.dataset.cluster_labels)))

        # Confusion matrix and per-cluster performance
        self.cm = confusion_matrix(
            self.true_clusters,
            self.predicted_clusters,
            labels=all_labels,  # Force full matrix shape
        )

        precision, recall, f1, _ = precision_recall_fscore_support(
            self.true_clusters,
            self.predicted_clusters,
            labels=all_labels,
            average=None,
            zero_division=0,
        )

        self.df_perf = pd.DataFrame(
            {
                "Cluster": all_labels,
                "Precision": precision,
                "Recall": recall,
                "F1": f1,
            }
        )

        logging.info(
            f"✅ Evaluation complete. Accuracy: {self.accuracy:.4f}, MSE: {self.embedding_mse:.6f}, Predicted: {predicted_cluster}"
        )

        self.true_next_cluster_labels = self.true_clusters
        self.predicted_next_cluster_labels = self.predicted_clusters

        # Save key single-step predictions
        self.predicted_next_embedding = E_hat
        self.predicted_next_cluster = int(predicted_cluster)
        self.predicted_next_cluster_confidence = float(
            predicted_prob[0][predicted_cluster]
        )
        self.predicted_probabilities = predicted_prob[0]

        self.cluster_distribution_info = self._analyze_cluster_distribution()

    def _analyze_cluster_distribution(self) -> dict:
        """
        Analyze and return per-cluster distribution stats across train and val.

        Returns:
            dict: Contains list of per-cluster counts and any val-only cluster warnings.
        """
        train_clusters = self.dataset.Y_cluster
        val_clusters = self.dataset.Y_cluster_val
        if val_clusters is None:
            val_clusters = np.array([])
        else:
            val_clusters = np.asarray(val_clusters)

        train_counts = Counter(train_clusters)
        val_counts = Counter(val_clusters)

        # Use union of config-declared clusters and seen clusters
        observed = set(train_counts) | set(val_counts)
        config_clusters = set(range(self.config.n_clusters))
        all_clusters = sorted(observed | config_clusters)

        distribution = []
        val_only = []

        for c in all_clusters:
            train_count = train_counts.get(c, 0)
            val_count = val_counts.get(c, 0)
            total = train_count + val_count
            distribution.append(
                {
                    "Cluster": int(c),
                    "Train Count": train_count,
                    "Val Count": val_count,
                    "Total Count": total,
                }
            )
            if train_count == 0 and val_count > 0:
                val_only.append(int(c))

        if val_only:
            logging.warning(f"🚨 Val-only clusters (missing in train): {val_only}")
        else:
            logging.info("✅ All clusters represented in training set.")

        return {"distribution": distribution, "val_only_clusters": val_only}

    def get_summary(self) -> dict:
        """
        Returns summary statistics for the latest evaluation in a compact dictionary.

        Returns:
            dict: Human-readable summary with model info, loss, and prediction confidence.
        """
        predicted_pct = f"{self.predicted_next_cluster_confidence * 100:.2f}%"

        return {
            "Model Type": self.forecast_model.name,
            "Output Normalization": "Enabled",  # toggle dynamically if needed
            "Epochs Run": self.epochs_run,
            "Final Train Loss": f"{self.final_train_loss:.6f}",
            "Final Val Loss": (
                f"{self.final_val_loss:.6f}" if self.final_val_loss else "N/A"
            ),
            "Predicted Cluster": self.predicted_next_cluster,
            "Predicted Cluster Probability": self.predicted_next_cluster_confidence,
            "Predicted Cluster Confidence": predicted_pct,
            "Predicted Probabilities": self.predicted_probabilities.tolist(),
            "Val-only Clusters": self.cluster_distribution_info.get(
                "val_only_clusters", []
            ),
        }

    def get_metrics(self) -> dict:
        """
        Returns all evaluation metrics for quantitative inspection or downstream use.

        Returns:
            dict: Includes MSE, accuracy, confusion matrix, and per-cluster stats.
        """
        return {
            "embedding_mse": self.embedding_mse,
            "classifier_accuracy": self.accuracy,
            "confusion_matrix": self.cm.tolist(),
            "performance": self.df_perf.to_dict(orient="records"),
            "cluster_distribution": self.cluster_distribution_info.get(
                "distribution", []
            ),
        }
