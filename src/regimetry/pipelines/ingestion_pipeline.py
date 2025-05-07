from regimetry.services.data_ingestion_service import DataIngestionService
from regimetry.logger_manager import LoggerManager

logging = LoggerManager.get_logger(__name__)

class IngestionPipeline:
    """
    Orchestrates the supervised data ingestion workflow:
    - Loads raw data
    - Applies preprocessing and feature engineering
    - Classifies targets
    - Splits and saves train/val/test datasets
    - Logs metadata and file locations
    """

    def __init__(self, val_size: float = 0.2, test_size: float = 0.2):
        self.val_size = val_size
        self.test_size = test_size

    def run(self):
        logging.info("🚀 Starting full supervised ingestion pipeline...")

        ingestion_service = DataIngestionService()

        train_path, val_path, test_path, feature_metadata = ingestion_service.initiate_data_ingestion(
            val_size=self.val_size,
            test_size=self.test_size
        )

        logging.info("✅ Ingestion pipeline completed.")
        logging.info(f"Train set saved to: {train_path}")
        logging.info(f"Validation set saved to: {val_path}")
        logging.info(f"Test set saved to: {test_path}")
        logging.info(f"Identified features: {feature_metadata}")

        return {
            "train_path": train_path,
            "val_path": val_path,
            "test_path": test_path,
            "features": feature_metadata
        }
