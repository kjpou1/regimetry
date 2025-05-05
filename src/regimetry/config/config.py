import os

import yaml
from dotenv import load_dotenv

from regimetry.models import SingletonMeta


class Config(metaclass=SingletonMeta):
    _is_initialized = False

    def __init__(self):
        if Config._is_initialized:
            return

        load_dotenv()

        self._debug = os.getenv("DEBUG", False)
        self._config_path = os.getenv("CONFIG_PATH", "config/default.yaml")
        self._model_type = None
        self._best_of_all = False
        self._save_best = False
        self._batch_size = int(os.getenv("BATCH_SIZE", 5))
        self._max_batches = os.getenv("MAX_BATCHES")
        self._max_batches = int(self._max_batches) if self._max_batches else None
        self._batch_offset = int(os.getenv("BATCH_OFFSET", 0))

        self.BASE_DIR = os.getenv("BASE_DIR", "artifacts")
        self.RAW_DATA_DIR = os.path.join(self.BASE_DIR, "data", "raw")
        self.MODEL_DIR = os.path.join(self.BASE_DIR, "models")
        self.MODEL_FILE_PATH = os.path.join(self.MODEL_DIR, "model.pkl")
        self.PREPROCESSOR_FILE_PATH = os.path.join(self.BASE_DIR, "preprocessor.pkl")
        self.LOG_DIR = os.path.join(self.BASE_DIR, "logs")
        self.HISTORY_DIR = os.path.join(self.BASE_DIR, "history")
        self.HISTORY_FILE_PATH = os.path.join(self.HISTORY_DIR, "training_history.json")
        self.REPORTS_DIR = os.path.join(self.BASE_DIR, "reports")
        self.PROCESSED_DATA_DIR = os.path.join(self.BASE_DIR, "data", "processed")

        self._signal_data_dir = os.getenv("SIGNAL_DATA_DIR", "artifacts/data/raw")

        self._ensure_directories_exist()
        Config._is_initialized = True

    def _ensure_directories_exist(self):
        for d in [
            self.RAW_DATA_DIR,
            self.PROCESSED_DATA_DIR,
            self.MODEL_DIR,
            self.LOG_DIR,
            self.REPORTS_DIR,
            self.HISTORY_DIR,
        ]:
            os.makedirs(d, exist_ok=True)

    def load_from_yaml(self, path: str):
        """
        Override config values from a YAML config file.
        Logs changes to config values.
        """
        if not os.path.exists(path):
            print(f"[Config] YAML config file not found: {path}")
            return

        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        print(f"[Config] Loaded YAML config: {path}")

        if "debug" in data:
            print(f"[Config] Overriding 'debug': {self._debug} → {data['debug']}")
            self.debug = data["debug"]

        if "signal_data_dir" in data:
            print(f"[Config] Overriding 'signal_data_dir': {self._signal_data_dir} → {data['signal_data_dir']}")
            self.signal_data_dir = data["signal_data_dir"]

    @property
    def config_path(self):
        return self._config_path

    @config_path.setter
    def config_path(self, value):
        if not isinstance(value, str):
            raise ValueError("config_path must be a string.")
        self._config_path = value

    @property
    def debug(self):
        return self._debug

    @debug.setter
    def debug(self, value):
        if not isinstance(value, bool):
            raise ValueError("debug must be a boolean.")
        self._debug = value

    @property
    def signal_data_dir(self):
        return self._signal_data_dir

    @signal_data_dir.setter
    def signal_data_dir(self, value):
        if not isinstance(value, str):
            raise ValueError("SIGNAL_DATA_DIR must be a string.")
        self._signal_data_dir = value

    @classmethod
    def initialize(cls):
        if not cls._is_initialized:
            cls()

    @classmethod
    def is_initialized(cls):
        return cls._is_initialized

    @classmethod
    def reset(cls):
        cls._is_initialized = False
        cls._instances = {}
