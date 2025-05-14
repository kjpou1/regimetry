import os

import yaml
from dotenv import load_dotenv

from regimetry.models import SingletonMeta
from regimetry.utils.path_utils import get_project_root
from regimetry.utils.path_utils import ensure_all_dirs_exist


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

        
        self.PROJECT_ROOT = get_project_root()
        # Resolve BASE_DIR intelligently
        base_dir_env = os.getenv("BASE_DIR", "artifacts")
        if os.path.isabs(base_dir_env):
            self.BASE_DIR = base_dir_env
        else:
            # Resolve relative to project root (assumed to be two levels up from this file)
            self.BASE_DIR = os.path.join(self.PROJECT_ROOT, base_dir_env)

        self.RAW_DATA_DIR = os.path.join(self.BASE_DIR, "data", "raw")
        self.MODEL_DIR = os.path.join(self.BASE_DIR, "models")
        self.MODEL_FILE_PATH = os.path.join(self.MODEL_DIR, "model.pkl")
        self.PREPROCESSOR_FILE_PATH = os.path.join(self.BASE_DIR, "preprocessor.pkl")
        self.LOG_DIR = os.path.join(self.BASE_DIR, "logs")
        self.HISTORY_DIR = os.path.join(self.BASE_DIR, "history")
        self.HISTORY_FILE_PATH = os.path.join(self.HISTORY_DIR, "training_history.json")
        self.REPORTS_DIR = os.path.join(self.BASE_DIR, "reports")
        self.PROCESSED_DATA_DIR = os.path.join(self.BASE_DIR, "data", "processed")
        self.EMBEDDINGS_DIR = os.path.join(self.BASE_DIR, "embeddings")
        self.TRANSFORMER_DIR = os.path.join(self.BASE_DIR, "transformer")

        self._signal_input_path = os.getenv("SIGNAL_INPUT_PATH", os.path.join(self.RAW_DATA_DIR, "signal_input.csv"))
        self._output_name = os.getenv("OUTPUT_NAME", "embeddings.npy")
        self._rhd_threshold = 0.002

        # Default to include all columns and exclude none
        self._include_columns = "*"  # Default to include all columns
        self._exclude_columns = []  # Empty list to exclude specific columns

        self._window_size = os.getenv("WINDOW_SIZE", 30)
        self._stride = os.getenv("STRIDE", 1)

        self._encoding_method = os.getenv("ENCODING_METHOD", "sinusoidal")
        self._encoding_style = os.getenv("ENCODING_STYLE", "interleaved")
        self._embedding_dim = os.getenv("EMBEDDING_DIM", None)  # or your default

        self._report_format = os.getenv("REPORT_FORMAT", ["matplotlib", "plotly"])  # Default to both
        if isinstance(self._report_format, str):
            # Convert comma-separated string to list if needed
            self._report_format = [fmt.strip() for fmt in self._report_format.split(",") if fmt.strip()]

        self._report_palette = os.getenv("REPORT_PALETTE", "tab10")  # Default seaborn/mpl palette

        self._ensure_directories_exist()
        Config._is_initialized = True

    def _ensure_directories_exist(self):
        ensure_all_dirs_exist([
            self.RAW_DATA_DIR,
            self.PROCESSED_DATA_DIR,
            self.EMBEDDINGS_DIR,
            self.TRANSFORMER_DIR,
            self.MODEL_DIR,
            self.LOG_DIR,
            self.REPORTS_DIR,
            self.HISTORY_DIR,
        ])

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

        if "signal_input_path" in data:
            print(f"[Config] Overriding 'signal_input_path': {self._signal_input_path} → {data['signal_input_path']}")
            self.signal_input_path = data["signal_input_path"]

        # Set attributes from the loaded YAML file
        if "include_columns" in data:
            print(f"[Config] Overriding 'include_columns': {self._include_columns} → {data['include_columns']}")
            self.include_columns = data["include_columns"]
        if "exclude_columns" in data:
            print(f"[Config] Overriding 'exclude_columns': {self._exclude_columns} → {data['exclude_columns']}")
            self.exclude_columns = data["exclude_columns"]
        if "output_name" in data:
            print(f"[Config] Overriding 'output_name': {self._output_name} → {data['output_name']}")
            self.output_name = data["output_name"]
        if "embedding_path" in data:
            print(f"[Config] Overriding 'embedding_path': {data['embedding_path']}")
            self.embedding_path = data["embedding_path"]

        if "regime_data_path" in data:
            print(f"[Config] Overriding 'regime_data_path': {data['regime_data_path']}")
            self.regime_data_path = data["regime_data_path"]

        if "output_dir" in data:
            print(f"[Config] Overriding 'output_dir': {data['output_dir']}")
            self.output_dir = data["output_dir"]

        if "window_size" in data:
            print(f"[Config] Overriding 'window_size': {data['window_size']}")
            self.window_size = int(data["window_size"])

        if "n_clusters" in data:
            print(f"[Config] Overriding 'n_clusters': {data['n_clusters']}")
            self.n_clusters = int(data["n_clusters"])

        if "stride" in data:
            print(f"[Config] Overriding 'stride': {data['stride']}")
            self.stride = int(data["stride"])

        if "encoding_method" in data:
            print(f"[Config] Overriding 'encoding_method': {data['encoding_method']}")
            self.encoding_method = data["encoding_method"]

        if "encoding_style" in data:
            print(f"[Config] Overriding 'encoding_style': {data['encoding_style']}")
            self.encoding_style = data["encoding_style"]

        if "embedding_dim" in data:
            print(f"[Config] Overriding 'embedding_dim': {data['embedding_dim']}")
            self.embedding_dim = int(data["embedding_dim"])
                            
        if "report_format" in data:
            fmt = data["report_format"]
            if isinstance(fmt, list):
                print(f"[Config] Overriding 'report_format': {self._report_format} → {fmt}")
                self.report_format = fmt
            else:
                raise ValueError("report_format must be a list of strings like ['matplotlib', 'plotly']")

        if "report_palette" in data:
            print(f"[Config] Overriding 'report_palette': {data['report_palette']}")
            self.report_palette = data["report_palette"]
            
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
    def signal_input_path(self) -> str:
        val = getattr(self, "_signal_input_path", None)
        return self._resolve_path(val)

    @signal_input_path.setter
    def signal_input_path(self, value: str):
        if not isinstance(value, str):
            raise ValueError("signal_input_path must be a string.")
        self._signal_input_path = value

    @property
    def rhd_threshold(self) -> float:
        return self._rhd_threshold

    @rhd_threshold.setter
    def rhd_threshold(self, value: float):
        self._rhd_threshold = value


    @property
    def include_columns(self):
        """
        Getter for include_columns.
        """
        return self._include_columns

    @include_columns.setter
    def include_columns(self, value):
        """
        Setter for include_columns.
        If value is "*" (default), all columns are included.
        """
        if value == "*":
            self._include_columns = "*"
        elif isinstance(value, list):
            self._include_columns = value
        else:
            raise ValueError("Include columns must be a list or '*'")

    @property
    def exclude_columns(self):
        """
        Getter for exclude_columns.
        """
        return self._exclude_columns

    @exclude_columns.setter
    def exclude_columns(self, value):
        """
        Setter for exclude_columns.
        Expects a list of columns to exclude.
        """
        if isinstance(value, list):
            self._exclude_columns = value
        else:
            raise ValueError("Exclude columns must be a list")
        
    @property
    def output_name(self) -> str:
        return self._output_name

    @output_name.setter
    def output_name(self, value: str):
        if not isinstance(value, str):
            raise ValueError("output_name must be a string.")
        self._output_name = value

    @property
    def embedding_path(self) -> str:
        val = getattr(self, "_embedding_path", None)
        return self._resolve_path(val)

    @embedding_path.setter
    def embedding_path(self, value: str):
        self._embedding_path = value

    @property
    def regime_data_path(self) -> str:
        val = getattr(self, "_regime_data_path", None)
        return self._resolve_path(val)

    @regime_data_path.setter
    def regime_data_path(self, value: str):
        self._regime_data_path = value
    
    @property
    def output_dir(self) -> str:
        val = getattr(self, "_output_dir", None)
        return self._resolve_path(val)

    @output_dir.setter
    def output_dir(self, value: str):
        self._output_dir = value

    @property
    def window_size(self) -> int:
        return getattr(self, "_window_size", 30)

    @window_size.setter
    def window_size(self, value: int):
        self._window_size = int(value)

    @property
    def n_clusters(self) -> int:
        return getattr(self, "_n_clusters", 3)

    @n_clusters.setter
    def n_clusters(self, value: int):
        self._n_clusters = int(value)

    @property
    def window_size(self) -> int:
        return getattr(self, "_window_size", 30)

    @window_size.setter
    def window_size(self, value: int):
        self._window_size = int(value)

    @property
    def stride(self) -> int:
        return getattr(self, "_stride", 1)

    @stride.setter
    def stride(self, value: int):
        self._stride = int(value)

    @property
    def encoding_method(self) -> str:
        return self._encoding_method

    @encoding_method.setter
    def encoding_method(self, value: str):
        self._encoding_method = value

    @property
    def encoding_style(self) -> str:
        return self._encoding_style

    @encoding_style.setter
    def encoding_style(self, value: str):
        self._encoding_style = value

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    @embedding_dim.setter
    def embedding_dim(self, value: int):
        self._embedding_dim = int(value)

    @property
    def report_format(self) -> list[str]:
        return self._report_format

    @report_format.setter
    def report_format(self, value: list[str]):
        if not isinstance(value, list):
            raise ValueError("report_format must be a list")
        for fmt in value:
            if fmt not in ["matplotlib", "plotly"]:
                raise ValueError(f"Unsupported report format: {fmt}")
        self._report_format = value

    @property
    def report_palette(self) -> str:
        return self._report_palette

    @report_palette.setter
    def report_palette(self, value: str):
        if not isinstance(value, str):
            raise ValueError("report_palette must be a string.")
        self._report_palette = value

    def _resolve_path(self, val: str) -> str:
        if not val:
            return None
        if os.path.isabs(val):
            return val
        # Tier 1: try resolving relative to BASE_DIR
        base_resolved = os.path.join(self.BASE_DIR, val)
        if os.path.exists(base_resolved):
            print(f"[Config] Resolved (BASE_DIR): {val} → {base_resolved}")
            return base_resolved
        # Tier 2: try resolving relative to PROJECT_ROOT
        root_resolved = os.path.join(self.PROJECT_ROOT, val)
        if os.path.exists(root_resolved):
            print(f"[Config] Resolved (PROJECT_ROOT): {val} → {root_resolved}")
            return root_resolved
        # Fallback: assume BASE_DIR anyway
        fallback = base_resolved
        print(f"[Config] Resolved (fallback to BASE_DIR): {val} → {fallback}")
        return fallback

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
