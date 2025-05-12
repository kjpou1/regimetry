import os
from dataclasses import dataclass

from regimetry.config.config import Config

@dataclass
class DataTransformationConfig:
    """
    Configuration for data transformation.
    Defines file paths for transformed data and the preprocessor object.
    """

    config: Config = Config()

    # Input data (no split — entire dataset used)
    input_data_path: str = os.path.join(
        config.PROCESSED_DATA_DIR, "regime_input.csv"
    )

    # Output: where to save the fitted preprocessor (e.g., a pipeline.pkl)
    transformer_object_path: str = os.path.join(
        config.TRANSFORMER_DIR, "preprocessor.pkl"
    )
