import os

import yaml
from dotenv import load_dotenv

from regimetry.models import SingletonMeta
from regimetry.utils.path_utils import ensure_all_dirs_exist, get_project_root


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
        self.BASELINE_METADATA_DIR = os.path.join(self.BASE_DIR, "baseline_metadata")
        self.FORECAST_MODEL_DIR = os.path.join(self.BASE_DIR, "forecast_models")

        self._signal_input_path = os.getenv(
            "SIGNAL_INPUT_PATH", os.path.join(self.RAW_DATA_DIR, "signal_input.csv")
        )
        self._rhd_threshold = 0.002

        # Default to include all columns and exclude none
        self._include_columns = "*"  # Default to include all columns
        self._exclude_columns = []  # Empty list to exclude specific columns

        self._window_size = os.getenv("WINDOW_SIZE", None)
        self._stride = os.getenv("STRIDE", None)

        self._encoding_method = os.getenv("ENCODING_METHOD", "sinusoidal")
        self._encoding_style = os.getenv("ENCODING_STYLE", "interleaved")
        self._embedding_dim = os.getenv("EMBEDDING_DIM", None)

        self._head_size = int(os.getenv("HEAD_SIZE", 256))
        self._num_heads = int(os.getenv("NUM_HEADS", 4))
        self._ff_dim = int(os.getenv("FF_DIM", 128))
        self._num_transformer_blocks = int(os.getenv("NUM_TRANSFORMER_BLOCKS", 2))
        self._dropout = float(os.getenv("DROPOUT", 0.1))

        self._instrument = os.getenv("INSTRUMENT", "Unknown")

        self._output_name = os.getenv("OUTPUT_NAME", f"{self.experiment_id}.npy")

        self._report_format = os.getenv(
            "REPORT_FORMAT", ["matplotlib", "plotly"]
        )  # Default to both
        if isinstance(self._report_format, str):
            # Convert comma-separated string to list if needed
            self._report_format = [
                fmt.strip() for fmt in self._report_format.split(",") if fmt.strip()
            ]

        self._report_palette = os.getenv(
            "REPORT_PALETTE", "tab10"
        )  # Default seaborn/mpl palette

        self._report_font_path = os.getenv(
            "REPORT_FONT_PATH", "./assets/DejaVuSans.ttf"
        )

        self._deterministic = (
            os.getenv("REGIMETRY_DETERMINISTIC", "true").lower() == "true"
        )
        self._random_seed = int(os.getenv("REGIMETRY_RANDOM_SEED", 42))

        self._embedding_dir = os.getenv("EMBEDDING_DIR", None)
        self._cluster_assignment_path = os.getenv("CLUSTER_ASSIGNMENT_PATH", None)
        self._model_type = os.getenv("MODEL_TYPE", None)
        self._training_profile_path = os.getenv(
            "TRAINING_PROFILE_PATH", "./configs/default_training_profile.yaml"
        )
        self._n_neighbors = os.getenv("N_NEIGHBORS", None)

        self._base_config = os.getenv("BASE_CONFIG", None)
        self._baseline_metadata_dir = os.getenv(
            "BASELINE_METADATA_DIR", self.BASELINE_METADATA_DIR
        )

        self._ensure_directories_exist()
        Config._is_initialized = True

    def _ensure_directories_exist(self):
        ensure_all_dirs_exist(
            [
                self.RAW_DATA_DIR,
                self.PROCESSED_DATA_DIR,
                self.EMBEDDINGS_DIR,
                self.TRANSFORMER_DIR,
                self.MODEL_DIR,
                self.LOG_DIR,
                self.REPORTS_DIR,
                self.HISTORY_DIR,
                self.BASELINE_METADATA_DIR,
            ]
        )

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
            print(
                f"[Config] Overriding 'signal_input_path': {self._signal_input_path} → {data['signal_input_path']}"
            )
            self.signal_input_path = data["signal_input_path"]

        # Set attributes from the loaded YAML file
        if "include_columns" in data:
            print(
                f"[Config] Overriding 'include_columns': {self._include_columns} → {data['include_columns']}"
            )
            self.include_columns = data["include_columns"]
        if "exclude_columns" in data:
            print(
                f"[Config] Overriding 'exclude_columns': {self._exclude_columns} → {data['exclude_columns']}"
            )
            self.exclude_columns = data["exclude_columns"]
        if "output_name" in data:
            print(
                f"[Config] Overriding 'output_name': {self._output_name} → {data['output_name']}"
            )
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

        if "report_palette" in data:
            print(f"[Config] Overriding 'report_palette': {data['report_palette']}")
            self.report_palette = data["report_palette"]

        if "head_size" in data:
            print(f"[Config] Overriding 'head_size': {data['head_size']}")
            self.head_size = int(data["head_size"])

        if "num_heads" in data:
            print(f"[Config] Overriding 'num_heads': {data['num_heads']}")
            self.num_heads = int(data["num_heads"])

        if "ff_dim" in data:
            print(f"[Config] Overriding 'ff_dim': {data['ff_dim']}")
            self.ff_dim = int(data["ff_dim"])

        if "num_transformer_blocks" in data:
            print(
                f"[Config] Overriding 'num_transformer_blocks': {data['num_transformer_blocks']}"
            )
            self.num_transformer_blocks = int(data["num_transformer_blocks"])

        if "dropout" in data:
            print(f"[Config] Overriding 'dropout': {data['dropout']}")
            self.dropout = float(data["dropout"])

        if "report_format" in data:
            fmt = data["report_format"]
            if isinstance(fmt, list):
                print(
                    f"[Config] Overriding 'report_format': {self._report_format} → {fmt}"
                )
                self.report_format = fmt
            else:
                raise ValueError(
                    "report_format must be a list of strings like ['matplotlib', 'plotly']"
                )

        if "instrument" in data:
            print(f"[Config] Overriding 'instrument': {data['instrument']}")
            self.instrument = data["instrument"]

        if "deterministic" in data:
            print(f"[Config] Overriding 'deterministic': {data['deterministic']}")
            self.deterministic = bool(data["deterministic"])

        if "random_seed" in data:
            print(
                f"[Config] Overriding 'random_seed': {self._random_seed} → {data['random_seed']}"
            )
            self.set_random_seed(int(data["random_seed"]))

        if "cluster_assignment_path" in data:
            print(
                f"[Config] Overriding 'cluster_assignment_path': {data['cluster_assignment_path']}"
            )
            self.cluster_assignment_path = data["cluster_assignment_path"]

        if "model_type" in data:
            print(f"[Config] Overriding 'model_type': {data['model_type']}")
            self.model_type = data["model_type"]

        if "n_neighbors" in data:
            print(f"[Config] Overriding 'n_neighbors': {data['n_neighbors']}")
            self.n_neighbors = int(data["n_neighbors"])

        if "embedding_dir" in data:
            print(f"[Config] Overriding 'embedding_dir': {data['embedding_dir']}")
            self.embedding_dir = data["embedding_dir"]

        if "baseline_metadata_dir" in data:
            print(
                f"[Config] Overriding 'baseline_metadata_dir': {data['baseline_metadata_dir']}"
            )
            self.baseline_metadata_dir = data["baseline_metadata_dir"]

        if "training_profile_path" in data:
            print(
                f"[Config] Overriding 'training_profile_path': {data['training_profile_path']}"
            )
            self.training_profile_path = data["training_profile_path"]

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
    def head_size(self) -> int:
        return self._head_size

    @head_size.setter
    def head_size(self, value: int):
        self._head_size = int(value)

    @property
    def num_heads(self) -> int:
        return self._num_heads

    @num_heads.setter
    def num_heads(self, value: int):
        self._num_heads = int(value)

    @property
    def ff_dim(self) -> int:
        return self._ff_dim

    @ff_dim.setter
    def ff_dim(self, value: int):
        self._ff_dim = int(value)

    @property
    def num_transformer_blocks(self) -> int:
        return self._num_transformer_blocks

    @num_transformer_blocks.setter
    def num_transformer_blocks(self, value: int):
        self._num_transformer_blocks = int(value)

    @property
    def dropout(self) -> float:
        return self._dropout

    @dropout.setter
    def dropout(self, value: float):
        self._dropout = float(value)

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

    @property
    def instrument(self) -> str:
        return self._instrument

    @instrument.setter
    def instrument(self, value: str):
        if not isinstance(value, str):
            raise ValueError("instrument must be a string.")
        self._instrument = value

    @property
    def deterministic(self) -> bool:
        return self._deterministic

    @deterministic.setter
    def deterministic(self, value: bool):
        if not isinstance(value, bool):
            raise ValueError("deterministic must be a boolean.")
        self._deterministic = value

    def get_random_seed(self):
        return self._random_seed

    def set_random_seed(self, value):
        if not isinstance(value, int):
            raise ValueError("random_seed must be an integer")
        self._random_seed = value

    @property
    def cluster_assignment_path(self) -> str:
        val = getattr(self, "_cluster_assignment_path", None)
        return self._resolve_path(val)

    @cluster_assignment_path.setter
    def cluster_assignment_path(self, value: str):
        if not isinstance(value, str):
            raise ValueError("cluster_assignment_path must be a string.")
        self._cluster_assignment_path = value

    @property
    def model_type(self) -> str:
        return self._model_type

    @model_type.setter
    def model_type(self, value: str):
        if not isinstance(value, str):
            raise ValueError("model_type must be a string.")
        self._model_type = value

    @property
    def n_neighbors(self) -> int:
        return getattr(self, "_n_neighbors", 5)

    @n_neighbors.setter
    def n_neighbors(self, value: int):
        self._n_neighbors = int(value)

    @property
    def embedding_dir(self) -> str:
        val = getattr(self, "_embedding_dir", None)
        return self._resolve_path(val)

    @embedding_dir.setter
    def embedding_dir(self, value: str):
        if not isinstance(value, str):
            raise ValueError("embedding_dir must be a string.")
        self._embedding_dir = value

    @property
    def base_config(self) -> str:
        return self._base_config

    @base_config.setter
    def base_config(self, value: str):
        if not isinstance(value, str):
            raise ValueError("base_config must be a string.")
        self._base_config = value

    @property
    def embedding_file(self) -> str:
        return os.path.join(self.embedding_dir, "embedding.npy")

    @property
    def embedding_metadata_path(self) -> str:
        return os.path.join(self.embedding_dir, "embedding_metadata.json")

    @property
    def baseline_metadata_dir(self) -> str:
        val = getattr(self, "_baseline_metadata_dir", None)
        return self._resolve_path(val)

    @baseline_metadata_dir.setter
    def baseline_metadata_dir(self, value: str):
        if not isinstance(value, str):
            raise ValueError("baseline_metadata_dir must be a string path")
        self._baseline_metadata_dir = value

    @property
    def experiment_id(self) -> str:
        method = self.encoding_method.lower()
        enc = "sin" if method.startswith("sin") else "learn"
        dim = self.embedding_dim

        parts = [
            self.instrument,
            f"ws{self.window_size}",
            f"{enc}{dim}" if dim is not None else enc,
        ]

        # Only append encoding style if sinusoidal
        if enc == "sin":
            parts.append(self.encoding_style)

        parts.append(f"nc{self.n_clusters}")
        return "_".join(parts)

    @property
    def report_font_path(self) -> str:
        return self._resolve_path(self._report_font_path)

    @property
    def training_profile_path(self) -> str:
        return self._resolve_path(self._training_profile_path)

    @training_profile_path.setter
    def training_profile_path(self, value: str):
        if not isinstance(value, str):
            raise ValueError("training_profile_path must be a string.")
        self._training_profile_path = value

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
