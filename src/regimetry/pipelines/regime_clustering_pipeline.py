"""
RegimeClusteringPipeline

This pipeline performs unsupervised market regime detection using precomputed
transformer embeddings. It applies clustering, dimensionality reduction, 
and generates rich visual reports to help interpret structural market behaviors.

Steps:
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

logging = LoggerManager.get_logger(__name__)


class RegimeClusteringPipeline:
    def __init__(self):
        """
        Initializes the clustering pipeline using configuration values.
        Prepares input paths and ensures the output directory exists.
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
        Executes the full clustering pipeline:
        - Loads embeddings
        - Applies clustering
        - Reduces dimensions
        - Aligns and saves cluster-labeled data
        - Generates visualization reports
        """
        logging.info("🚀 Starting regime clustering pipeline")

        # STEP 1: Load embeddings
        embeddings = np.load(self.embedding_path)
        logging.info(f"📥 Loaded embeddings: shape={embeddings.shape}")

        # STEP 2: Standardize the embeddings before clustering
        scaler = StandardScaler()
        embeddings_scaled = scaler.fit_transform(embeddings)

        # STEP 3: Apply Spectral Clustering to discover regime clusters
        spectral = SpectralClustering(
            n_clusters=self.n_clusters,
            affinity='nearest_neighbors',
            assign_labels='kmeans',
            random_state=42,
        )
        cluster_labels = spectral.fit_predict(embeddings_scaled)
        logging.info("🔗 Spectral clustering complete.")

        # STEP 4: Reduce dimensions for visualization
        tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
        tsne_coords = tsne.fit_transform(embeddings_scaled)

        umap = UMAP(n_components=2, random_state=42)
        umap_coords = umap.fit_transform(embeddings_scaled)

        # STEP 5: Load original regime data and attach cluster labels
        df = pd.read_csv(self.regime_data_path)
        df['Cluster_ID'] = np.nan

        # Align clusters starting from (window_size - 1) index
        df['Cluster_ID'] = pd.Series(
            cluster_labels,
            index=range(self.window_size - 1, self.window_size - 1 + len(cluster_labels))
        )
        df['Cluster_ID'] = df['Cluster_ID'].astype('Int64')  # Preserve NaNs

        # STEP 6: Save merged regime-labeled dataset
        cluster_path = os.path.join(self.output_dir, "cluster_assignments.csv")
        df.to_csv(cluster_path, index=False)
        logging.info(f"💾 Cluster assignments saved: {cluster_path}")

        # STEP 7: Generate scatter/timeline/overlay plots
        report_service = ClusteringReportService()
        report_service.generate_all(
            df=df,
            cluster_labels=cluster_labels,
            tsne_coords=tsne_coords,
            umap_coords=umap_coords,
        )

        return df
