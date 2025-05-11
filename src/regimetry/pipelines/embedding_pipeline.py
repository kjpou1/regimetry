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

class EmbeddingPipeline:
    """
    Embedding pipeline to transform raw data into windowed transformer embeddings.
    """

    def __init__(self, val_size: float = 0.2, test_size: float = 0.2):
        self.config = Config()
        self.test_size = test_size
        self.val_size = val_size
        self.data_ingestion_service = DataIngestionService()
        self.data_transformation_service = DataTransformationService()

    def run_pipeline(self):
        try:
            logging.info("🚀 Starting embedding pipeline.")

            # STEP 1: Data Ingestion
            logging.info("📥 Running data ingestion service.")
            train_path, val_path, test_path, feature_metadata = self.data_ingestion_service.initiate_data_ingestion(
                val_size=self.val_size,
                test_size=self.test_size
            )

            # STEP 2: Data Transformation
            logging.info("🔄 Running data transformation.")
            train_arr, _, preprocessor_obj = self.data_transformation_service.initiate_data_transformation()
            logging.info("✅ Data transformed.")

            if hasattr(train_arr, "toarray"):
                train_arr = train_arr.toarray()

            # STEP 3: Rolling Window Generation
            window_gen = RollingWindowGenerator(data=train_arr, window_size=30)
            rolling_windows = window_gen.generate()
            logging.info(f"🧊 Rolling windows shape: {rolling_windows.shape}")

            # STEP 4: Positional Encoding
            X = tf.convert_to_tensor(rolling_windows, dtype=tf.float32)
            X_pe_final = PositionalEncoding.add(
                X,
                method='sinusoidal',  # or 'learnable'
                encoding_style='stacked'  # or 'interleaved'
            )
            logging.info(f"📐 Positional encoding applied: shape={X_pe_final.shape}")

            # STEP 5: Embedding Extraction (optional, add model here)
            # encoder = build_unsupervised_transformer_encoder(...)
            # embeddings = encoder.predict(X_pe_final, batch_size=64)
            # logging.info(f"✅ Embeddings shape: {embeddings.shape}")

            return {
                "ingested_shape": train_arr.shape,
                "rolling_windows_shape": rolling_windows.shape,
                "encoded_shape": X_pe_final.shape,
                # "embeddings": embeddings,
            }

        except Exception as e:
            logging.error(f"❌ Error in embedding pipeline: {e}")
            raise CustomException(e, sys) from e

