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
from regimetry.services.model_embedding_service import ModelEmbeddingService
from regimetry.utils.file_utils import save_array
from regimetry.utils.utils import inspect_transformed_output, print_feature_block

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
            logging.info(f"📁 Ingested paths: train={train_path}, val={val_path}, test={test_path}")

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

            # STEP 5: Embedding Extraction
            service = ModelEmbeddingService(input_shape=X_pe_final.shape[1:])
            embeddings = service.embed(X_pe_final)
            logging.info(f"✅ Embeddings shape: {embeddings.shape}")

            filename = self.config.output_name
            filepath = os.path.join(self.config.EMBEDDINGS_DIR, filename)
            save_array(embeddings, filepath)
            logging.info(f"✅ Embeddings saved: {filepath}")


            
            return {
                "ingested_shape": train_arr.shape,
                "rolling_windows_shape": rolling_windows.shape,
                "encoded_shape": X_pe_final.shape,
                "embeddings": embeddings,
            }

        except Exception as e:
            logging.error(f"❌ Error in embedding pipeline: {e}")
            raise CustomException(e, sys) from e

