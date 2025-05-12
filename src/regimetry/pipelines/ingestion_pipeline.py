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

        output_path, feature_metadata = ingestion_service.initiate_data_ingestion()

        logging.info("✅ Ingestion pipeline completed.")
        logging.info(f"Full dataset saved to: {output_path}")
        logging.info(f"Identified features: {feature_metadata}")

        return {
            "output_path": output_path,
            "features": feature_metadata
        }
