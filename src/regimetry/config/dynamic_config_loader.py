import os
from pathlib import Path
from typing import Dict, Optional

import yaml

from regimetry.config.config import Config


class DynamicConfigLoader:
    """
    Loads an instrument-specific base config and injects dynamic runtime parameters.
    Automatically derives standard artifact paths (embedding, report, cluster output).
    Optionally creates required output directories when instructed.
    """

    def __init__(self, config_dir: str = "configs"):
        """
        Parameters:
        ----------
        config_dir : str
            Directory where base YAML configs are stored (e.g., 'configs/')
        artifacts_dir : str
            Root directory for all generated output artifacts
        """
        self.config = Config()
        self.config_dir = Path(config_dir)
        self.artifacts_dir = self.config.BASE_DIR

    def load(
        self,
        instrument: str,
        window_size: int,
        stride: int,
        encoding_method: str,
        embedding_dim: int,
        n_clusters: int,
        encoding_style: Optional[str] = None,
        export_path: Optional[str] = None,
        create_dirs: bool = False,
        force: bool = False,
        clean: bool = False,
    ) -> Dict:
        """
        Load a base YAML config and inject runtime parameters, including resolved output paths.

        Parameters:
        ----------
        instrument : str
            Forex instrument symbol (e.g., "EUR_USD")
        window_size : int
            Rolling window size used during embedding
        stride : int
            Stride between windows
        encoding_method : str
            One of "sinusoidal" or "learnable"
        embedding_dim : int
            Dimensionality of the positional encoding
        n_clusters : int
            Number of clusters for spectral clustering
        encoding_style : Optional[str]
            Positional encoding style (e.g., "interleaved", "stacked")
        export_path : Optional[str]
            If set, exports the final config to this path as a YAML file
        create_dirs : bool
            If True, creates directories for embedding and report output paths
        force : bool
            If True, forcibly creates output directories even if they exist
        clean : bool
            If True, removes existing output directories before creation

        Returns:
        -------
        config : Dict
            Fully merged and updated configuration dictionary
        """
        # Try to resolve base config from Config object
        base_config = self.config.base_config

        if base_config:
            base_path = Path(base_config)
            print(f"[Loader] Using explicitly provided base_config: {base_path}")
        else:
            base_path = self.config_dir / f"{instrument}_base.yaml"
            print(f"[Loader] Using default instrument base_config: {base_path}")

        if not base_path.exists():
            raise FileNotFoundError(f"Base config not found: {base_path}")

        with open(base_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        # 🔧 Inject dynamic values
        config["instrument"] = instrument
        config["window_size"] = window_size
        config["stride"] = stride
        config.setdefault("positional_encoding", {})
        config["positional_encoding"]["type"] = encoding_method
        config["positional_encoding"]["dim"] = embedding_dim
        if encoding_style:
            config["positional_encoding"]["style"] = encoding_style

        config.setdefault("clustering", {})
        config["clustering"]["n_clusters"] = n_clusters

        # 🧠 Auto-generate standardized relative output paths
        dim_key = f"{embedding_dim}"
        style_key = encoding_style or "default"
        posenc_key = f"{encoding_method}{dim_key}"

        embedding_rel_path = (
            Path("embeddings")
            / instrument
            / f"ws{window_size}"
            / f"{posenc_key}_{style_key}"
            / "embedding.npy"
        )
        report_rel_path = (
            Path("reports")
            / instrument
            / f"ws{window_size}"
            / posenc_key
            / style_key
            / f"nc{n_clusters}"
        )
        cluster_rel_path = report_rel_path / "cluster_assignments.csv"

        # 🛠 Resolve full output paths
        embedding_full_path = self.config._resolve_path(
            self.artifacts_dir / embedding_rel_path
        )
        report_full_path = self.config._resolve_path(
            self.artifacts_dir / report_rel_path
        )
        cluster_full_path = self.config._resolve_path(
            self.artifacts_dir / cluster_rel_path
        )

        # 🔥 Optionally clean old directories
        if clean:
            import shutil

            if embedding_full_path.parent.exists():
                shutil.rmtree(embedding_full_path.parent, ignore_errors=True)
            if report_full_path.exists():
                shutil.rmtree(report_full_path, ignore_errors=True)

        # 🛠 Create directories if requested or forced
        if create_dirs or force:
            embedding_full_path.parent.mkdir(parents=True, exist_ok=True)
            report_full_path.mkdir(parents=True, exist_ok=True)

        # 📦 Save all paths as resolved (Config will resolve again internally if needed)
        config["embedding_path"] = str(embedding_full_path)
        config["report_dir"] = str(report_full_path)
        config["cluster_output_path"] = self.config._resolve_path(
            str(cluster_full_path)
        )
        config["regime_data_path"] = self.config._resolve_path(
            str(Path("data/processed/regime_input.csv"))
        )
        config["output_dir"] = self.config._resolve_path(str(report_full_path))

        # 📝 Optionally export merged config for inspection
        if export_path:
            with open(export_path, "w") as f:
                yaml.dump(config, f)
            print(f"📝 Exported merged config to: {export_path}")

        return config
