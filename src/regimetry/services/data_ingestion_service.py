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
            df = TrendSignalService.add_trend_signals(df, self.config.rhd_threshold)
            logging.info("Added trend signal flags.")

            # Drop missing values based on the included columns (if '*' include all columns)
            # df = df.dropna(subset=included_columns)
            # logging.info(f"Dropped rows with missing values in columns: {included_columns}")
            # **Handle missing values for the critical columns (LC_Prev, LP_Prev, LP_Slope)**
            missing_columns = ['LC_Prev', 'LP_Prev', 'LP_Slope']
            df[missing_columns] = df[missing_columns].fillna(0)  # Fill NaN with zero
            logging.info(f"Filled NaN values with 0 for columns: {missing_columns}")
    
            # List of columns to fill NaN values with 'Flat'
            columns_to_fill = ['Prevailing_Trend','Baseline_Aligned', 'Trend_Agreement', 'Entry_Trigger', 'Entry_Confirmed']

            # Fill NaN values in these columns with 'Flat'
            df[columns_to_fill] = df[columns_to_fill].fillna('Flat')


            if df.isnull().all().sum() > 0:
                raise DataProcessingError("All rows contain NaNs after processing!")

            # Excluding columns based on config
            if self.config.exclude_columns:
                df = df.drop(columns=self.config.exclude_columns, axis=1)
                logging.info(f"Excluding columns: {self.config.exclude_columns}")

            # Including only specified columns if config is provided
            if self.config.include_columns == "*":  # Handle the '*' case
                logging.info("Including all columns")
                included_columns = df.columns.tolist()  # Include all columns
            else:
                included_columns = [
                    col for col in self.config.include_columns if col in df.columns
                ]
                df = df[included_columns]  # Only keep the specified columns
                logging.info(f"Including columns: {included_columns}")


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

    def initiate_data_ingestion(self):
        """
        Orchestrates the data ingestion process:
        - Reads the input data.
        - Applies preprocessing (renaming columns, dropping unnecessary fields).
        - Identifies features.
        - Saves the processed full dataset (no split).
        - Returns path and feature metadata.
        """
        logging.info("Entered the data ingestion method.")
        try:
            if not os.path.exists(self.config.signal_input_path):
                logging.error(f"Input file not found at {self.config.signal_input_path}")
                raise FileNotFoundError(f"File not found: {self.config.signal_input_path}")

            # Step 1: Load the raw input
            df = pd.read_csv(self.config.signal_input_path)
            logging.info("Read the dataset as a pandas DataFrame.")

            # Step 2: Preprocess
            df_cleaned = self.preprocess_data(df)

            # Step 3: Feature extraction
            feature_metadata = self.identify_features(df_cleaned)

            # Step 4: Save the full processed dataset
            output_path = self.ingestion_config.processed_data_path
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            df_cleaned.to_csv(output_path, index=False, header=True)
            logging.info(f"Full dataset saved at: {output_path}")

            return output_path, feature_metadata

        except Exception as e:
            raise CustomException(e, sys) from e

