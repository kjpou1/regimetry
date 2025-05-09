import os
from dataclasses import dataclass

from regimetry.config.config import Config


@dataclass
class DataTransformationConfig:
    """
    Configuration for data transformation.
    Defines the file path for saving the preprocessor object.
    """

    config: Config = Config()  # Access the centralized Config singleton

    # File paths
    train_data_path: str = os.path.join(
        config.PROCESSED_DATA_DIR, "train.csv"
    )  # Training data
    validation_data_path: str = os.path.join(
        config.PROCESSED_DATA_DIR, "validation.csv"
    )  # Testing data
