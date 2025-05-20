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
from regimetry.services.clustering_report_service import ClusteringReportService
from regimetry.utils.cluster_utils import attach_cluster_labels, verify_cluster_alignment
from regimetry.services.analysis_prompt_service import AnalysisPromptService

logging = LoggerManager.get_logger(__name__)


class RegimeClusteringPipeline:
    def __init__(self):
        """
        Initializes the clustering pipeline using configuration values.
        Sets up input/output paths and ensures output directory exists.
        """
        self.config = Config()

        self.embedding_path = self.config.embedding_path
        self.regime_data_path = self.config.regime_data_path
        self.output_dir = self.config.output_dir
        self.window_size = self.config.window_size
        self.n_clusters = self.config.n_clusters

        os.makedirs(self.output_dir, exist_ok=True)

    def run(self):
        """
        Executes the full regime clustering pipeline:
        - Loads and scales embeddings
        - Applies Spectral Clustering
        - Computes dimensionality reductions (t-SNE, UMAP)
        - Attaches cluster labels to original data
        - Saves labeled dataset and visualizations
        """
        logging.info("🚀 Starting regime clustering pipeline")

        # STEP 1: Load embeddings
        embeddings = np.load(self.embedding_path)
        logging.info(f"📥 Loaded embeddings: shape={embeddings.shape}")

        # STEP 1.5: Validate embedding alignment with expected window size
        regime_df = pd.read_csv(self.regime_data_path)
        expected_embedding_len = len(regime_df) - self.config.window_size + 1

        if embeddings.shape[0] != expected_embedding_len:
            raise ValueError(
                f"[❌] Embedding window mismatch:\n"
                f"    → Config window_size = {self.config.window_size}\n"
                f"    → Regime input rows = {len(regime_df)}\n"
                f"    → Expected embeddings = {expected_embedding_len}\n"
                f"    → Found embeddings = {embeddings.shape[0]}\n\n"
                f"💡 Re-run embedding generation with correct window_size or fix the config."
            )

        # STEP 2: Standardize embeddings
        scaler = StandardScaler()
        embeddings_scaled = scaler.fit_transform(embeddings)

        # STEP 3: Spectral Clustering
        spectral = SpectralClustering(
            n_clusters=self.n_clusters,
            affinity='nearest_neighbors',
            assign_labels='kmeans',
            random_state=42,
        )
        cluster_labels = spectral.fit_predict(embeddings_scaled)
        logging.info("🔗 Spectral clustering complete.")

        # STEP 4: Dimensionality reduction
        tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
        tsne_coords = tsne.fit_transform(embeddings_scaled)
        logging.info(f"📉 t-SNE complete: shape={tsne_coords.shape}")

        umap = UMAP(n_components=2, random_state=42)
        umap_coords = umap.fit_transform(embeddings_scaled)
        logging.info(f"🌀 UMAP complete: shape={umap_coords.shape}")

        # STEP 4.5: Save t-SNE and UMAP coordinates
        tsne_path_npy = os.path.join(self.output_dir, "tsne_coords.npy")
        umap_path_npy = os.path.join(self.output_dir, "umap_coords.npy")
        np.save(tsne_path_npy, tsne_coords)
        np.save(umap_path_npy, umap_coords)
        logging.info(f"💾 Saved t-SNE coords: {tsne_path_npy}")
        logging.info(f"💾 Saved UMAP coords: {umap_path_npy}")

        # Optional: also save as CSV for external inspection
        tsne_path_csv = os.path.join(self.output_dir, "tsne_coords.csv")
        umap_path_csv = os.path.join(self.output_dir, "umap_coords.csv")

        pd.DataFrame(tsne_coords, columns=["x", "y"]).assign(Cluster_ID=cluster_labels).to_csv(tsne_path_csv, index=False)
        pd.DataFrame(umap_coords, columns=["x", "y"]).assign(Cluster_ID=cluster_labels).to_csv(umap_path_csv, index=False)

        logging.info(f"📄 t-SNE CSV saved: {tsne_path_csv}")
        logging.info(f"📄 UMAP CSV saved: {umap_path_csv}")

        # STEP 5: attach cluster labels
        regime_df = attach_cluster_labels(regime_df, cluster_labels, window_size=self.config.window_size)
        verify_cluster_alignment(regime_df, window_size=self.config.window_size)

        # STEP 6: Save labeled dataset
        cluster_path = os.path.join(self.output_dir, "cluster_assignments.csv")
        regime_df.to_csv(cluster_path, index=False)
        logging.info(f"💾 Cluster assignments saved: {cluster_path}")

        # STEP 7: Generate reports
        report_service = ClusteringReportService()
        report_service.generate_all(
            df=regime_df,
            cluster_labels=cluster_labels,
            tsne_coords=tsne_coords,
            umap_coords=umap_coords,
        )

        # STEP 8: Save analysis prompt

        analysis_prompt_service = AnalysisPromptService() 
        analysis_prompt = analysis_prompt_service.get_prompt()       

        prompt_filename = f"{self.config.experiment_id}_analysis_prompt.md"
        prompt_path = os.path.join(self.output_dir, prompt_filename)

        with open(prompt_path, "w", encoding="utf-8") as f:
            f.write(analysis_prompt)

        logging.info(f"📝 Analysis prompt saved: {prompt_path}")

        return regime_df
