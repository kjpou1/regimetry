import json
import os
from datetime import datetime, timezone
from os import path

import joblib
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import normalize
from tensorflow.keras.callbacks import ModelCheckpoint

from regimetry.config.config import Config
from regimetry.config.training_profile_config import TrainingProfileConfig
from regimetry.logger_manager import LoggerManager
from regimetry.services.forecast.classifier_trainer_service import (
    ForecastClassifierTrainerService,
)
from regimetry.services.forecast.dataset_service import ForecastDatasetService
from regimetry.services.forecast.evaluation_service import ForecastEvaluationService
from regimetry.services.forecast.model_builder_service import (
    ForecastModelBuilderService,
)
from regimetry.services.forecast.report_service import ForecastReportService

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

        self.training_profile = TrainingProfileConfig.from_yaml(
            self.config.training_profile_path
            if self.config.training_profile_path
            else "./configs/default_training_profile.yaml"
        )

        self.output_dir = self.config.output_dir
        # Output location
        if self.output_dir is None:
            if not self.config.instrument:
                raise ValueError(
                    "Instrument is required if output-dir is not specified."
                )
            self.output_dir = self.config.output_dir or os.path.join(
                self.config.FORECAST_MODEL_DIR, self.config.instrument
            )
            self.config.output_dir = self.output_dir

        os.makedirs(self.output_dir, exist_ok=True)

    def run(self):
        logging.info("🚀 Starting forecast training pipeline")

        # STEP 1–2: Load embeddings and build training dataset
        dataset = self.dataset_service.build_dataset(
            validation_split=(
                self.training_profile.validation_split
                if self.training_profile.use_validation
                else 0.0
            )
        )

        X = dataset.X
        Y = dataset.Y
        Y_cluster = dataset.Y_cluster
        embeddings = dataset.embeddings
        cluster_labels = dataset.cluster_labels

        logging.info(dataset.summary())

        # STEP 3: Build forecast model using profile
        model_builder = ForecastModelBuilderService(self.training_profile)

        model, callbacks = model_builder.build_model(
            input_shape=(self.config.window_size, embeddings.shape[1]),
            output_dim=embeddings.shape[1],
            normalize_output=self.training_profile.normalize_output,
        )

        logging.info(f"🧠 Training model: {self.training_profile.model_type}")
        model.summary()

        # ⚙️ Prepare training parameters using values from the training profile
        fit_kwargs = {
            "x": dataset.X,  # 🧠 Input training data: shape (N, window_size, embedding_dim)
            "y": dataset.Y,  # 🎯 Target embeddings: shape (N, embedding_dim)
            "epochs": self.training_profile.epochs,  # ⏳ Total training epochs
            "batch_size": self.training_profile.batch_size,  # 📦 Mini-batch size from training profile
            "callbacks": callbacks,  # 🛎️ Callbacks (e.g. EarlyStopping, LR schedulers)
            "verbose": self.training_profile.verbose,  # 📣 Log training progress per epoch
            "shuffle": False,  # 🔒 Important: maintain time order for sequential models
        }

        # 🔍 Validation logic: use precomputed val split if available
        if self.training_profile.use_validation and dataset.X_val is not None:
            # ✅ Better than validation_split — you explicitly constructed this in ForecastDatasetService
            # ✅ Ensures val set is a true chronological holdout
            # ❌ Using validation_split would randomly carve out val data, violating time series ordering
            fit_kwargs["validation_data"] = (dataset.X_val, dataset.Y_val)
        else:
            # 🚫 Disable internal split if we're not using precomputed validation
            fit_kwargs["validation_split"] = 0.0

        # 🚀 Begin model training
        history = model.fit(**fit_kwargs)

        lr_schedule = model.optimizer.learning_rate
        final_learning_rate = (
            float(lr_schedule.numpy()) if hasattr(lr_schedule, "numpy") else lr_schedule
        )
        if hasattr(lr_schedule, "numpy"):
            print(f"🔄 Final learning rate (pre-training): {final_learning_rate:.6f}")
        else:
            print(f"🔄 Learning rate schedule: {lr_schedule}")

        model_path = os.path.join(self.output_dir, "embedding_forecaster.keras")
        model.save(model_path)

        logging.info(f"💾 Forecaster model saved: {model_path}")
        best_model_path = None
        if self.training_profile.checkpoint_enabled:
            for cb in callbacks:
                if isinstance(cb, ModelCheckpoint):
                    best_model_path = cb.filepath
                    if isinstance(
                        best_model_path, str
                    ):  # some TF versions may use callables
                        if os.path.exists(best_model_path):
                            logging.info(
                                f"🏆 Best model checkpoint saved at: {best_model_path}"
                            )
                        else:
                            logging.warning(
                                f"⚠️ Best model checkpoint expected but not found at: {best_model_path}"
                            )

        # STEP 4: Train KNN classifier on original embeddings
        valid_mask = ~np.isnan(cluster_labels)
        X_knn = embeddings[valid_mask]
        y_knn = cluster_labels[valid_mask].astype(int)
        # Resolve n_neighbors priority: CLI/Config override > training profile
        n_neighbors = self.config.n_neighbors or self.training_profile.n_neighbors

        classifier_trainer = ForecastClassifierTrainerService(n_neighbors=n_neighbors)
        _, classifier_output_path = classifier_trainer.train(X_knn, y_knn)

        # # STEP 5: Save all artifacts
        summary = {
            "instrument": self.config.instrument,
            "model_type": self.training_profile.model_type,
            "loss": self.training_profile.loss,
            "embedding_dim": embeddings.shape[1],
            "window_size": self.config.window_size,
            "stride": self.config.stride,
            "normalize_output": self.training_profile.normalize_output,
            "epochs": self.training_profile.epochs,
            "learning_rate": self.training_profile.learning_rate,
            "final_learning_rate": round(float(final_learning_rate), 6),
            "use_validation": self.training_profile.use_validation,
            "early_stop": self.training_profile.early_stopping,
            "lr_scheduler": self.training_profile.lr_scheduler,
            "checkpoint": self.training_profile.model_checkpoint,
            "n_samples_used": X.shape[0],
            "n_clusters": int(np.nanmax(cluster_labels) + 1),
            "n_neighbors": n_neighbors,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        summary_path = os.path.join(self.output_dir, "training_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        logging.info(f"📄 Training summary saved: {summary_path}")
        logging.info("✅ Forecast training pipeline completed.")

        # 🧪 STEP 6: Save training history for loss curve
        history_path = os.path.join(self.output_dir, "forecast_history.json")
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(history.history, f)
        logging.info(f"📈 Training history saved: {history_path}")

        # 🧪 STEP 7: Run evaluation
        evaluator = ForecastEvaluationService(
            forecast_model_path=model_path,
            best_model_path=best_model_path,
            classifier_model_path=classifier_output_path,
            history_path=history_path,
            dataset=dataset,
        )

        evaluator.evaluate()

        eval_summary = evaluator.get_summary()
        eval_metrics = evaluator.get_metrics()

        # Save evaluation metrics
        metrics_path = os.path.join(self.output_dir, "evaluation_metrics.json")
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(eval_metrics, f, indent=2)
        logging.info(f"📊 Evaluation metrics saved: {metrics_path}")

        # Print summary
        # Build a pretty multi-line string first
        summary_lines = ["\n📋 Forecast Model Summary:"]
        for k, v in eval_summary.items():
            summary_lines.append(f"{k:30}: {v}")

        # Join and log once
        logging.info("\n".join(summary_lines))

        eval_summary_path = os.path.join(self.output_dir, "evaluation_summary.json")
        with open(eval_summary_path, "w", encoding="utf-8") as f:
            json.dump(eval_summary, f, indent=2)
        logging.info(f"📊 Evaluation summary saved: {eval_summary_path}")

        # 🧾 STEP 8: Generate forecast evaluation report visuals
        report_service = ForecastReportService.from_evaluator(
            evaluator=evaluator, history_path=history_path, output_dir=self.output_dir
        )
        report_service.generate_all()
        logging.info("📊 Forecast evaluation report visuals saved.")
