from dataclasses import dataclass
from typing import Optional

@dataclass
class CommandLineArgs:
    """
    Structured command-line arguments for the regimetry pipeline.

    Arguments are grouped into:
    - CLI command selection
    - File/config overrides
    - Embedding parameters
    - Clustering options
    - Positional encoding behavior
    """

    # === Core CLI ===
    command: str                     # Subcommand: 'ingest', 'embed', or 'cluster'
    config: Optional[str]           # Optional path to YAML config file
    debug: bool                     # Enable verbose logging

    # === File Paths & General Overrides ===
    signal_input_path: Optional[str] = None   # Path to input CSV of market features
    output_name: Optional[str] = None         # Optional override for output .npy filename

    # === Embedding & Clustering Shared Args ===
    embedding_path: Optional[str] = None      # Path to saved embedding .npy file
    regime_data_path: Optional[str] = None    # Path to signal-enriched regime input CSV
    output_dir: Optional[str] = None          # Directory to store outputs and visualizations
    window_size: Optional[int] = None         # Rolling window size used for embedding
    stride: Optional[int] = None              # Stride between rolling windows
    n_clusters: Optional[int] = None          # Number of clusters for spectral clustering

    # === Positional Encoding Overrides ===
    encoding_method: Optional[str] = None     # 'sinusoidal' or 'learnable'
    encoding_style: Optional[str] = None      # 'interleaved' or 'stacked' (only for sinusoidal)
    embedding_dim: Optional[int] = None       # Required if method is 'learnable'

    # === Interpretability-Specific Args ===
    input_path: Optional[str] = None          # Path to cluster-labeled CSV (used by interpret)
    cluster_col: Optional[str] = "Cluster_ID" # Column in CSV that holds regime labels
    save_heatmap: bool = False                # Whether to save transition heatmap PNG
    save_csv: bool = False                    # Whether to save decision table + matrix to CSV
    save_json: bool = False                   # Whether to generate regime metadate json file

    instrument: Optional[str] = None
    base_config: Optional[str] = None
    create_dir: bool = False
    force: bool = False
    clean: bool = False    