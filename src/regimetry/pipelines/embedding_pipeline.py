import os
import random
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
from regimetry.utils.file_utils import save_array, save_embedding_metadata
from regimetry.utils.utils import inspect_transformed_output, print_feature_block

import tensorflow as tf

logging = LoggerManager.get_logger(__name__)

# Ensure deterministic behavior
def set_deterministic(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass  # May not exist in older TF versions

class EmbeddingPipeline:
    """
    Embedding pipeline to transform raw data into windowed transformer embeddings.
    """

    def __init__(self):
        self.config = Config()
        self.window_size = self.config.window_size
        self.stride = self.config.stride
        self.data_ingestion_service = DataIngestionService()
        self.data_transformation_service = DataTransformationService()
        seed = self.config.get_random_seed()
        set_deterministic(seed=seed)

    def run_pipeline(self):
        try:
            logging.info("🚀 Starting embedding pipeline.")

            # STEP 1: Data Ingestion
            logging.info("📥 Running data ingestion service.")
            # Use the single 'regime_input.csv' file for all
            full_data_path, feature_metadata = self.data_ingestion_service.initiate_data_ingestion()
            logging.info(f"📁 Ingested full dataset: {full_data_path}")

            # STEP 2: Data Transformation
            logging.info("🔄 Running data transformation.")
            # No validation split here, just use the full dataset
            full_data_arr, preprocessor_obj = self.data_transformation_service.initiate_data_transformation()
            logging.info("✅ Data transformed.")

            if hasattr(full_data_arr, "toarray"):
                full_data_arr = full_data_arr.toarray()

            # STEP 3: Rolling Window Generation
            window_gen = RollingWindowGenerator(data=full_data_arr, window_size=self.window_size, stride=self.stride)
            rolling_windows = window_gen.generate()
            logging.info(f"🧊 Rolling windows shape: {rolling_windows.shape}")

            # STEP 4: Positional Encoding
            X = tf.convert_to_tensor(rolling_windows, dtype=tf.float32)

            # Optional projection to match embedding_dim
            if self.config.embedding_dim and X.shape[-1] != self.config.embedding_dim:
                logging.info(f"🔧 Projecting input features from {X.shape[-1]} → {self.config.embedding_dim}")
                X = tf.keras.layers.Dense(self.config.embedding_dim)(X)

            X_pe_final = PositionalEncoding.add(
                X,
                method=self.config.encoding_method,
                encoding_style=self.config.encoding_style,
                learnable_dim=self.config.embedding_dim
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

            # Save embedding metadata (alongside the .npy file)
            metadata_path = save_embedding_metadata(
                embeddings=embeddings,
                output_path=filepath,
                features_used=preprocessor_obj.get_feature_names_out().tolist(),
                window_size=self.window_size,
                stride=self.stride,
                encoding_method=self.config.encoding_method,
                encoding_style=self.config.encoding_style,
                embedding_model="UnsupervisedTransformerEncoder",
                source_file=self.config.signal_input_path,
            )
            logging.info(f"📄 Metadata saved: {metadata_path}")

            return {
                "ingested_shape": full_data_arr.shape,
                "rolling_windows_shape": rolling_windows.shape,
                "encoded_shape": X_pe_final.shape,
                "embeddings": embeddings,
            }

        except Exception as e:
            logging.error(f"❌ Error in embedding pipeline: {e}")
            raise CustomException(e, sys) from e


