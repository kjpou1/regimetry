import os
import sys
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import FunctionTransformer, Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from regimetry.config.config import Config
from regimetry.exception import CustomException
from regimetry.logger_manager import LoggerManager
from regimetry.models.data_transformation_config import DataTransformationConfig
from regimetry.utils.file_utils import save_object

logging = LoggerManager.get_logger(__name__)


class DataTransformationService:
    """
    Handles all data transformation logic, including:
    - Selecting valid columns based on config
    - Building and applying a preprocessing pipeline
    - Returning transformed arrays ready for model input (e.g., for transformer embedding)
    - Supports cyclical encoding for 'Hour'
    """

    def __init__(self):
        """
        Initializes the transformation service using project-wide configs.
        """
        self.config = Config()
        self.transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self, input_df: pd.DataFrame, use_cyclical_encoding=False) -> ColumnTransformer:
        """
        Builds a column-wise transformation pipeline for numerical and categorical data.
        Applies sine/cosine encoding to 'Hour' if specified.

        Args:
            input_df (pd.DataFrame): The dataset used to derive feature types and fit transformers.
            use_cyclical_encoding (bool): If True, encodes 'Hour' using sine/cosine.

        Returns:
            ColumnTransformer: The preprocessing object.
        """
        try:
            all_columns = input_df.columns.tolist()

            # Determine which columns to include
            if self.config.include_columns == "*":
                columns_to_include = all_columns
            else:
                columns_to_include = [col for col in self.config.include_columns if col in all_columns]

            # Remove excluded columns
            if self.config.exclude_columns:
                columns_to_include = [col for col in columns_to_include if col not in self.config.exclude_columns]

            logging.info(f"Including columns: {columns_to_include}")
            logging.info(f"Excluding columns: {self.config.exclude_columns}")

            # Identify feature types
            numerical_columns = [col for col in columns_to_include if input_df[col].dtype in ['float64', 'int64']]
            categorical_columns = [col for col in columns_to_include if input_df[col].dtype == 'object']

            # Handle cyclical encoding for 'Hour'
            hour_transform = None
            if 'Hour' in columns_to_include:
                if use_cyclical_encoding:
                    def cyclic_transform(X):
                        """
                        Converts hour-of-day into sine and cosine components to preserve cyclical nature.
                        Example: hour 0 and 23 should be near each other in feature space.
                        """
                        radians = 2 * np.pi * X / 24
                        return pd.DataFrame({
                            "Hour_sin": np.sin(radians).flatten(),
                            "Hour_cos": np.cos(radians).flatten()
                        })

                    hour_transform = ("hour_pipeline", Pipeline([
                        ("cyclic_transform", FunctionTransformer(cyclic_transform, validate=False)),
                        ("scaler", StandardScaler())
                    ]), ["Hour"])

                    # Remove Hour from normal processing
                    columns_to_include.remove("Hour")
                    if "Hour" in numerical_columns:
                        numerical_columns.remove("Hour")
                    if "Hour" in categorical_columns:
                        categorical_columns.remove("Hour")
                else:
                    categorical_columns.append("Hour")

            # Define transformation pipelines
            num_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ])

            cat_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("one_hot_encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
            ])

            # Combine all transformers
            transformers = [
                ("num_pipeline", num_pipeline, numerical_columns),
                ("cat_pipeline", cat_pipeline, categorical_columns)
            ]

            if hour_transform:
                transformers.append(hour_transform)

            preprocessor = ColumnTransformer(transformers)
            return preprocessor

        except Exception as e:
            raise CustomException(e, sys) from e

    def initiate_data_transformation(self, save=True):
        """
        Loads the full dataset, applies preprocessing, and returns transformed data and fitted preprocessor.

        Returns:
            Tuple:
                - np.ndarray: Transformed data array
                - ColumnTransformer: Fitted preprocessor object
        """
        try:
            # Load full dataset (regime_input.csv)
            df = pd.read_csv(self.transformation_config.input_data_path)
            logging.info("📁 Loaded full dataset for transformation.")

            # Create preprocessing pipeline based on data schema
            preprocessing_obj = self.get_data_transformer_object(df)

            # Fit and transform the entire dataset
            logging.info("⚙️ Fitting and transforming dataset.")
            transformed_array = preprocessing_obj.fit_transform(df)

            logging.info("✅ Full dataset transformation complete.")
            logging.info(f"🔢 Transformed shape: {transformed_array.shape}")

            # Save preprocessor if configured
            if save:
                save_object(
                    file_path=self.transformation_config.transformer_object_path,
                    obj=preprocessing_obj
                )
                logging.info(f"💾 Preprocessor saved to: {self.transformation_config.transformer_object_path}")

            return transformed_array, preprocessing_obj

        except Exception as e:
            raise CustomException(e, sys) from e
