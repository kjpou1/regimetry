import os
from dataclasses import dataclass

from regimetry.config.config import Config

@dataclass
class DataIngestionConfig:
    """
    Configuration class for data ingestion.
    Defines the file paths for unified ingestion (no train/val/test split).
    """

    config: Config = Config()

    # Unified raw input file
    raw_data_path: str = os.path.join(config.RAW_DATA_DIR, "data.csv")

    # Single processed output file (full dataset, no split)
    processed_data_path: str = os.path.join(config.PROCESSED_DATA_DIR, "regime_input.csv")
