import os

import numpy as np
import pandas as pd
import umap
from sklearn.manifold import TSNE

from regimetry.config.config import Config
from regimetry.logger_manager import LoggerManager

logger = LoggerManager.get_logger(__name__)


class RegimeProjectionService:
    """
    Projects standardized embeddings into 2D using t-SNE and UMAP,
    using final remapped Cluster_IDs (post regime assignment).

    Outputs:
        - tsne_projection.npy / .csv
        - umap_projection.npy / .csv
    """

    def __init__(self):
        self.config = Config()
        self.output_dir = self.config.output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def run(
        self, embeddings: np.ndarray, cluster_ids: np.ndarray, seed: int = None
    ) -> dict:
        """
        Run t-SNE and UMAP projections on standardized embeddings.

        Args:
            embeddings (np.ndarray): Standardized embeddings of shape [n_samples, n_features].
            cluster_ids (np.ndarray): Final remapped Cluster_IDs of shape [n_samples].
            seed (int, optional): Random seed for reproducibility. Defaults to None.

        Returns:
            dict: {
                'tsne': np.ndarray of shape [n_samples, 2],
                'umap': np.ndarray of shape [n_samples, 2],
                'labels': np.ndarray of shape [n_samples]
            }
        """
        logger.info(f"📉 Running t-SNE and UMAP projections... (seed={seed})")

        tsne_proj = TSNE(
            n_components=2, perplexity=30, random_state=seed
        ).fit_transform(embeddings)
        umap_proj = umap.UMAP(n_components=2, random_state=seed).fit_transform(
            embeddings
        )

        return {
            "tsne": tsne_proj,
            "umap": umap_proj,
            "labels": cluster_ids,
        }

    def save(self, projections: dict):
        """
        Save projections to .npy and .csv. Assumes labels are already remapped.

        Args:
            projections (dict): Output from run()
        """
        for name in ["tsne", "umap"]:
            coords = projections[name]
            labels = projections["labels"]

            # Defensive check for shape mismatch
            if len(coords) != len(labels):
                logger.warning(
                    f"⚠️ Projection length mismatch — truncating to min(len(coords)={len(coords)}, len(labels)={len(labels)})"
                )
                n = min(len(coords), len(labels))
                coords = coords[:n]
                labels = labels[:n]

            # Save .npy
            npy_path = os.path.join(self.output_dir, f"{name}_projection.npy")
            np.save(npy_path, coords)
            logger.info(f"💾 Saved {name.upper()} projection to {npy_path}")

            # Save .csv with aligned Cluster_IDs
            df = pd.DataFrame(
                coords, columns=[f"{name.upper()}_X", f"{name.upper()}_Y"]
            )
            df["Cluster_ID"] = labels

            csv_path = os.path.join(self.output_dir, f"{name}_projection.csv")
            df.to_csv(csv_path, index=False, encoding="utf-8")
            logger.info(f"📄 Saved {name.upper()} projection CSV to {csv_path}")
