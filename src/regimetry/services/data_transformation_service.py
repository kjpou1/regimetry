import os
import sys
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
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
    """

    def __init__(self):
        """
        Initializes the transformation service using project-wide configs.
        """
        self.config = Config()
        self.transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self, train_df: pd.DataFrame) -> ColumnTransformer:
        """
        Builds a column-wise transformation pipeline for numerical and categorical data.

        Args:
            train_df (pd.DataFrame): Dataset to derive feature types and fit transformers on.

        Returns:
            Tuple:
                - ColumnTransformer: The preprocessing object.
                - List[str]: The final list of included features.
        """
        try:
            all_columns = train_df.columns.tolist()

            # Determine which columns to include based on config
            if self.config.include_columns == "*":
                columns_to_include = all_columns
            else:
                columns_to_include = [col for col in self.config.include_columns if col in all_columns]

            # Remove excluded columns, if any
            if self.config.exclude_columns:
                columns_to_include = [col for col in columns_to_include if col not in self.config.exclude_columns]

            logging.info(f"Including columns: {columns_to_include}")
            logging.info(f"Excluding columns: {self.config.exclude_columns}")

            # Split into numeric and categorical columns
            numerical_columns = [col for col in columns_to_include if train_df[col].dtype in ['float64', 'int64']]
            categorical_columns = [col for col in columns_to_include if train_df[col].dtype == 'object']

            # Define pipeline for numerical features
            num_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ])

            # Define pipeline for categorical features
            cat_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("one_hot_encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
            ])

            # Combine into a column transformer
            preprocessor = ColumnTransformer(transformers=[
                ("num_pipeline", num_pipeline, numerical_columns),
                ("cat_pipeline", cat_pipeline, categorical_columns)
            ])

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys) from e

    def initiate_data_transformation(self):
        """
        Loads the train and validation datasets, applies preprocessing, and returns transformed arrays.

        Returns:
            Tuple:
                - np.ndarray: Transformed training data
                - np.ndarray: Transformed validation data
                - ColumnTransformer: The fitted preprocessing object
                - List[str]: Final set of features used
        """
        try:
            train_df = pd.read_csv(self.transformation_config.train_data_path)
            validation_df = pd.read_csv(self.transformation_config.validation_data_path)

            logging.info("📁 Loaded train and validation datasets.")
            logging.info("🔧 Creating preprocessing pipeline.")

            preprocessing_obj = self.get_data_transformer_object(train_df)

            logging.info("⚙️ Applying transformation to training and validation data.")

            input_feature_train_arr = preprocessing_obj.fit_transform(train_df)
            input_feature_validation_arr = preprocessing_obj.transform(validation_df)

            logging.info("✅ Data transformation complete.")
            logging.info(f"🔢 Train shape: {input_feature_train_arr.shape}")
            logging.info(f"🔢 Validation shape: {input_feature_validation_arr.shape}")

            return input_feature_train_arr, input_feature_validation_arr, preprocessing_obj

        except Exception as e:
            raise CustomException(e, sys) from e

    def initiate_data_transformation_for_test(
        self, test_df: pd.DataFrame, preprocessing_obj: ColumnTransformer
    ) -> np.ndarray:
        """
        Applies a saved preprocessor to transform new (e.g., test) data.

        Args:
            test_df (pd.DataFrame): The test dataset to transform.
            preprocessing_obj (ColumnTransformer): Pre-fitted preprocessing object.

        Returns:
            np.ndarray: Transformed test data.
        """
        try:
            logging.info("📥 Transforming test dataset.")
            input_feature_test_arr = preprocessing_obj.transform(test_df)
            logging.info("✅ Test data transformation complete.")
            return input_feature_test_arr

        except Exception as e:
            raise CustomException(e, sys) from e
