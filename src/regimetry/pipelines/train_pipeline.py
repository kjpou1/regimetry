import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

from regimetry.config.config import Config
from regimetry.exception import CustomException
from regimetry.logger_manager import LoggerManager
from regimetry.models.model_container import ModelContainer
from regimetry.models.positional_encoding import PositionalEncoding
from regimetry.models.rolling_windows_generator import RollingWindowGenerator
from regimetry.services.data_ingestion_service import DataIngestionService
from regimetry.services.data_transformation_service import DataTransformationService
from regimetry.services.model_training_service import ModelTrainingService
from regimetry.utils.utils import inspect_transformed_output, print_feature_block
# from src.services.report_service import ReportService
# from src.utils.file_utils import save_json, save_object
# from src.utils.history_utils import append_training_history, update_training_history
# from src.utils.yaml_loader import load_model_config
import tensorflow as tf

logging = LoggerManager.get_logger(__name__)


class TrainPipeline:
    """
    Main training pipeline for transforming market data, generating windowed inputs,
    and training models for regime detection or prediction.
    """

    def __init__(self):
        """
        Initializes all required services using the configured project settings.
        """
        self.data_ingestion_service = DataIngestionService()
        self.data_transformation_service = DataTransformationService()
        self.model_training_service = ModelTrainingService()
        self.config = Config()

    def run_pipeline(self):
        """
        Orchestrates the full training pipeline from transformation to window generation
        and (optionally) model training.

        Returns:
            dict: Model report containing training results (if model training is enabled).
        """
        try:
            # STEP 1: Data Transformation
            logging.info("Starting data transformation.")
            train_arr, validation_arr, preprocessor_obj = (
                self.data_transformation_service.initiate_data_transformation()
            )
            logging.info("✅ Data transformed.")

            # Inspect the transformed train array
            # inspect_transformed_output(preprocessor_obj, train_arr, max_rows=20)

            # print_feature_block(preprocessor_obj, train_arr, pattern="RHD", max_rows=10)
            # print_feature_block(preprocessor_obj, train_arr, pattern="num_pipeline", max_cols=50)
            # print_feature_block(preprocessor_obj, train_arr)

            # STEP 2: Rolling Window Generation (for unsupervised embedding or sequence models)
            logging.info("🧊 Generating rolling windows.")
            if hasattr(train_arr, "toarray"):  # Handle sparse matrix if returned
                train_arr = train_arr.toarray()
            window_gen = RollingWindowGenerator(
                data=train_arr,
                window_size=30
            )
            rolling_windows = window_gen.generate()
            logging.info(f"✅ Rolling windows generated: shape={rolling_windows.shape}")

            X = tf.convert_to_tensor(rolling_windows, dtype=tf.float32)
            print(f"X shape: {X.shape}")
            X_pe = PositionalEncoding.add(X, method='sinusoidal')
            

            # Add learnable positional encoding
            encoded_output = PositionalEncoding.add(X, method='learnable', learnable_dim=1273)

            # --- OPTIONAL: STEP 3 - Model Training (currently commented out) ---

            # Initialize model tracking dictionaries
            model_report = {}
            model_instances = {}

            # --- PLACEHOLDER for supervised model training logic ---
            # See commented section below for training and model selection steps.

            # # STEP 3: Load model config
            # model_configs = load_model_config(self.config.model_config_path)
            # models_to_train = self.resolve_model_list(model_configs)

            # for model_type in models_to_train:
            #     train_results = self.model_training_service.train_and_validate(
            #         model_type, train_arr, validation_arr
            #     )
            #     model_report[model_type] = {
            #         "performance_metrics": train_results["performance_metrics"],
            #         "best_params": train_results["best_params"],
            #         "best_val_accuracy": train_results["best_val_accuracy"],
            #         "validation_report": train_results["validation_report"],
            #     }
            #     model_instances[model_type] = {"model": train_results["model"]}

            # # STEP 4: Select Best Model by Validation Accuracy
            # best_model_name = max(model_report, key=lambda m: model_report[m]["performance_metrics"]["validation_accuracy"])
            # best_model_results = model_report[best_model_name]
            # best_model = model_instances[best_model_name]["model"]

            # # STEP 5: Save History
            # history_entry = {
            #     "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            #     "model": best_model_name,
            #     "performance_metrics": best_model_results["performance_metrics"],
            #     "best_params": best_model_results["best_params"],
            #     "best_val_accuracy": best_model_results["best_val_accuracy"],
            #     "validation_report": best_model_results["validation_report"],
            #     "model_report": model_report,
            # }
            # update_training_history(history_entry)

            # # STEP 6: Save Best Model
            # if self.config.save_best:
            #     best_model = ModelContainer(best_model, preprocessor_obj)
            #     best_model.save(self.config.MODEL_FILE_PATH)
            #     logging.info(f"✅ Saved best model to {self.config.MODEL_FILE_PATH}")

            return model_report  # Will be empty unless training section is activated

        except Exception as e:
            logging.error(f"❌ Error in training pipeline: {e}")
            raise CustomException(e, sys) from e
