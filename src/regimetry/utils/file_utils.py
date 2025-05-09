import json
import os
import sys

import dill  # Serialization library for Python objects
import numpy as np
import pandas as pd

from regimetry.config.config import Config
from regimetry.exception import CustomException
from regimetry.logger_manager import LoggerManager

# Initialize logger
logging = LoggerManager.get_logger(__name__)


def save_object(file_path: str, obj: object) -> None:
    """
    Save a Python object to the specified file path using the `dill` library.

    Args:
        file_path (str): The file path where the object will be saved. If directories in the path do not exist, they will be created.
        obj (object): The Python object to serialize and save.

    Raises:
        CustomException: If an error occurs during the saving process, wraps and raises the error with additional context.

    Example:
        >>> example_obj = {"key": "value"}
        >>> save_object("artifacts/example.pkl", example_obj)
    """
    try:
        # Extract the directory path from the given file path
        dir_path = os.path.dirname(file_path)

        # Ensure the directory exists; create it if it doesn't
        os.makedirs(dir_path, exist_ok=True)

        # Open the file in binary write mode and serialize the object using dill
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        # If an error occurs, raise a CustomException with the error details and system information
        raise CustomException(e, sys) from e


def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys) from e


def save_json(file_path: str, obj: object) -> None:
    """
    Saves a Python object as a JSON file.

    Args:
        file_path (str): The path where the JSON file will be saved.
        obj (object): The Python object to be serialized to JSON.

    Raises:
        CustomException: If the object cannot be serialized or if there's an issue writing to the file.
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=4)
        logging.info(f"JSON saved successfully to {file_path}")
    except TypeError as e:
        logging.error(f"Unable to serialize object to JSON. Error: {e}")
        raise CustomException(f"Unable to serialize object to JSON. Error: {e}") from e
    except OSError as e:
        logging.error(f"Failed to write JSON file to {file_path}. Error: {e}")
        raise CustomException(
            f"Failed to write JSON file to {file_path}. Error: {e}"
        ) from e


def save_json_safe(data, filepath):
    """Save JSON ensuring NumPy types are converted to native Python types."""

    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()  # Convert arrays to lists
        if isinstance(obj, (np.int64, np.int32)):
            return int(obj)  # Convert NumPy integers
        if isinstance(obj, (np.float64, np.float32)):
            return float(obj)  # Convert NumPy floats
        return obj  # Default return

    data_serializable = json.loads(json.dumps(data, default=convert))

    with open(filepath, "w") as f:
        json.dump(data_serializable, f, indent=4)


def save_training_artifacts(history, model_type, run_id):
    """
    Save training artifacts, including the training history and metadata,
    to a model-specific directory under the history directory.

    Args:
        history: The history object returned by `model.fit`.
        metadata: A dictionary containing metadata about the training run.
        model_type: Name of the model type (e.g., mobile, efficientnet).
        run_id: Unique identifier for this training run.

    Raises:
        CustomException: If there is an error during saving.
    """
    try:
        config = Config()
        # Create model-specific subdirectory
        model_history_dir = os.path.join(config.HISTORY_DIR, model_type)
        os.makedirs(model_history_dir, exist_ok=True)

        # Define file paths
        history_file = os.path.join(model_history_dir, f"history_{run_id}.json")

        # Save history
        save_json(history_file, history)
        logging.info(f"Training history saved to {history_file}")

    except Exception as e:
        logging.error(
            f"Failed to save training artifacts for model {model_type}: {str(e)}"
        )
        raise CustomException(f"Failed to save training artifacts: {str(e)}")
