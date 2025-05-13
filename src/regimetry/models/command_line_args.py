from dataclasses import dataclass
from typing import Optional

@dataclass
class CommandLineArgs:
    command: str
    config: Optional[str]
    debug: bool

    # Shared/optional overrides
    signal_input_path: Optional[str] = None
    output_name: Optional[str] = None

    # Clustering-specific args
    embedding_path: Optional[str] = None
    regime_data_path: Optional[str] = None
    output_dir: Optional[str] = None
    window_size: Optional[int] = None
    stride: Optional[int] = None
    n_clusters: int = 3
