"""
RegimeClusteringPipeline

This pipeline performs unsupervised market regime detection using precomputed
transformer embeddings. It applies clustering, dimensionality reduction,
and generates rich visual reports to help interpret structural market behaviors.

### Pipeline Steps:
1. Load embeddings
2. Standardize embeddings
3. Cluster via Spectral Clustering
4. Reduce dimensions via t-SNE and UMAP
5. Attach cluster labels to original data
6. Save merged regime-labeled CSV
7. Generate visualizations via ClusteringReportService
"""

import os

import numpy as np
import pandas as pd
from sklearn.cluster import SpectralClustering
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from umap import UMAP

from regimetry.config.config import Config
from regimetry.logger_manager import LoggerManager
from regimetry.services.analysis_prompt_service import AnalysisPromptService
from regimetry.services.clustering_report_service import ClusteringReportService
from regimetry.services.pdf_report_service import PDFReportService
from regimetry.services.regime_assignment_service import RegimeAssignmentService
from regimetry.utils.cluster_utils import (
    attach_cluster_labels,
    verify_cluster_alignment,
)

logging = LoggerManager.get_logger(__name__)


class RegimeClusteringPipeline:
    def __init__(self):
        """
        Initializes the clustering pipeline using configuration values.
        Sets up input/output paths and ensures output directory exists.
        """
        self.config = Config()

        self.seed = self.config.get_random_seed() if self.config.deterministic else None
        logging.info(
            f"{'🧬 Deterministic mode ON' if self.seed is not None else '🎲 Randomized run'} (seed={self.seed})"
        )

        self.embedding_path = self.config.embedding_path
        self.regime_data_path = self.config.regime_data_path
        self.output_dir = self.config.output_dir
        self.window_size = self.config.window_size
        self.n_clusters = self.config.n_clusters

        os.makedirs(self.output_dir, exist_ok=True)

    def run(self):
        """
        Executes the full regime clustering pipeline:
        1. Load and standardize embeddings
        2. Apply spectral clustering
        3. Reduce dimensions via t-SNE and UMAP
        4. Assign and align cluster labels
        5. Generate clustering reports and visualizations
        6. Save prompt and PDF analysis
        """
        logging.info("🚀 Starting regime clustering pipeline")

        # STEP 1: Load embeddings
        embeddings = np.load(self.embedding_path)
        logging.info(f"📥 Loaded embeddings: shape={embeddings.shape}")

        # STEP 1.5: Validate alignment
        regime_df = pd.read_csv(self.regime_data_path)
        expected_len = len(regime_df) - self.window_size + 1
        if embeddings.shape[0] != expected_len:
            raise ValueError(
                f"[❌] Embedding mismatch:\n"
                f"    → Expected: {expected_len}, Got: {embeddings.shape[0]}"
            )

        # STEP 2: Standardize
        scaler = StandardScaler()
        embeddings_scaled = scaler.fit_transform(embeddings)

        # STEP 3: Clustering
        spectral = SpectralClustering(
            n_clusters=self.n_clusters,
            affinity="nearest_neighbors",
            assign_labels="kmeans",
            random_state=self.seed,
            eigen_solver="arpack",
        )
        cluster_labels = spectral.fit_predict(embeddings_scaled)
        logging.info("🔗 Spectral clustering complete")

        # STEP 4: Dimensionality reduction
        tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=self.seed)
        tsne_coords = tsne.fit_transform(embeddings_scaled)

        umap = UMAP(n_components=2, random_state=self.seed)
        umap_coords = umap.fit_transform(embeddings_scaled)

        # Save t-SNE/UMAP results
        np.save(os.path.join(self.output_dir, "tsne_coords.npy"), tsne_coords)
        np.save(os.path.join(self.output_dir, "umap_coords.npy"), umap_coords)

        pd.DataFrame(tsne_coords, columns=["x", "y"]).assign(
            Cluster_ID=cluster_labels
        ).to_csv(os.path.join(self.output_dir, "tsne_coords.csv"), index=False)
        pd.DataFrame(umap_coords, columns=["x", "y"]).assign(
            Cluster_ID=cluster_labels
        ).to_csv(os.path.join(self.output_dir, "umap_coords.csv"), index=False)

        # STEP 5: Align & attach labels
        assignment_service = RegimeAssignmentService()
        regime_df = assignment_service.assign_and_align(regime_df, cluster_labels)

        # STEP 6: Report generation
        ClusteringReportService().generate_all(
            df=regime_df,
            cluster_labels=cluster_labels,
            tsne_coords=tsne_coords,
            umap_coords=umap_coords,
        )

        # STEP 7: Prompt
        prompt_text = AnalysisPromptService().get_prompt()
        with open(
            os.path.join(
                self.output_dir, f"{self.config.experiment_id}_analysis_prompt.md"
            ),
            "w",
            encoding="utf-8",
        ) as f:
            f.write(prompt_text)

        # STEP 8: PDF
        PDFReportService().generate_pdf()

        logging.info("🏁 Regime clustering pipeline complete.")
        return regime_df
