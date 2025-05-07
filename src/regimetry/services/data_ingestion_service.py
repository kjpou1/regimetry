import os
import sys

import numpy as np
import pandas as pd

from regimetry.config.config import Config
from regimetry.exception import CustomException
from regimetry.logger_manager import LoggerManager
from regimetry.models.data_ingestion_config import DataIngestionConfig
from regimetry.models.data_processing_error import DataProcessingError
from regimetry.services.data_split_service import DataSplitService
from regimetry.services.trend_signal_service import TrendSignalService

logging = LoggerManager.get_logger(__name__)


class DataIngestionService:
    """
    A class for handling the data ingestion process.
    Reads input data, applies initial preprocessing, identifies features, and splits it into train/test datasets.
    """

    def __init__(self):
        """
        Initializes the DataIngestionService with the configuration.
        """
        self.ingestion_config = DataIngestionConfig()
        self.data_split_service = DataSplitService()
        self.config = Config()

    def preprocess_data(self, df: pd.DataFrame):
        """
        Performs initial data cleaning, column renaming, and drops unnecessary columns.

        Args:
            df (pd.DataFrame): The raw dataset.

        Returns:
            pd.DataFrame: The cleaned dataset.
        """
        try:
            # Excluding columns based on config
            if self.config.exclude_columns:
                df = df.drop(columns=self.config.exclude_columns, axis=1)
                logging.info(f"Excluding columns: {self.config.exclude_columns}")

            # Including only specified columns if config is provided
            if self.config.include_columns != "*":
                included_columns = [col for col in self.config.include_columns if col in df.columns]
                df = df[included_columns]
                logging.info(f"Including columns: {included_columns}") 
                           
            df = TrendSignalService.add_trend_signals(df, self.config.rhd_threshold)
            logging.info("Added trend signal flags.")

            return df

        except DataProcessingError as e:
            logging.error(f"Data Processing Error: {e}")
            raise CustomException(e, sys) from e

        except Exception as e:
            raise CustomException(e, sys) from e

    def identify_features(self, df: pd.DataFrame):
        """
        Identifies numerical, categorical including boolean features as numerical.

        Args:
            df (pd.DataFrame): The input dataset.

        Returns:
            dict: A dictionary containing numerical and categorical features.
        """
        try:
            # Identifying numerical features (includes booleans as 0 or 1)
            numerical_features = df.select_dtypes(include=["number", "bool"]).columns.tolist()

            # Identifying categorical features (typically non-numeric, non-boolean)
            categorical_features = df.select_dtypes(include=["object"]).columns.tolist()

            # Handle boolean columns by converting them to binary (0/1)
            for column in numerical_features:
                if df[column].dtype == 'bool':  # Checking for boolean columns
                    df[column] = df[column].astype(int)

            # Logging the identified features
            logging.info("Numerical features identified: %s", numerical_features)
            logging.info("Categorical features identified: %s", categorical_features)

            return {
                "numerical": numerical_features,
                "categorical": categorical_features,
            }
        except Exception as e:
            raise CustomException(e, sys) from e

    def initiate_data_ingestion(self, val_size=0.2, test_size=0.2):
        """
        Orchestrates the data ingestion process:
        - Reads the input data.
        - Applies preprocessing (renaming columns, dropping unnecessary fields).
        - Identifies features.
        - Splits the data into train, validation, and test sets.
        - Saves the raw, train, validation, and test datasets.

        Args:
            val_size (float): Proportion of the train dataset to include in the validation split.
            test_size (float): Proportion of the dataset to include in the test split.

        Returns:
            tuple: Paths to the train, validation, and test dataset files, and identified features.

        Raises:
            CustomException: Custom exception if any error occurs during the process.
        """
        logging.info("Entered the data ingestion method.")
        try:
            if not os.path.exists(self.config.signal_input_path):
                logging.error(f"Input file not found at {self.config.signal_input_path}")
                raise FileNotFoundError(
                    f"File not found: {self.config.signal_input_path}"
                )

            df = pd.read_csv(self.config.signal_input_path)
            logging.info("Read the dataset as a pandas DataFrame.")

            # Apply preprocessing (renaming, dropping unnecessary columns)
            df_cleaned = self.preprocess_data(df)

            # Identify features
            feature_metadata = self.identify_features(df_cleaned)

            os.makedirs(
                os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True
            )

            df_cleaned.to_csv(
                self.ingestion_config.raw_data_path, index=False, header=True
            )
            logging.info("Raw dataset saved successfully.")

            train_set, val_set, test_set = self.data_split_service.sequential_split(
                df_cleaned, test_size=test_size, val_size=val_size
            )

            # Save the datasets
            train_set.to_csv(
                self.ingestion_config.train_data_path, index=False, header=True
            )
            val_set.to_csv(
                self.ingestion_config.validation_data_path, index=False, header=True
            )
            test_set.to_csv(
                self.ingestion_config.test_data_path, index=False, header=True
            )
            logging.info("Train, validation, and test datasets saved successfully.")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.validation_data_path,
                self.ingestion_config.test_data_path,
                feature_metadata,
            )
        except Exception as e:
            raise CustomException(e, sys) from e
