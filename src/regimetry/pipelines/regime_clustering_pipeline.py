import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import SpectralClustering
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from umap import UMAP

from regimetry.config.config import Config
from regimetry.logger_manager import LoggerManager

logging = LoggerManager.get_logger(__name__)


class RegimeClusteringPipeline:
    def __init__(self):
        self.config = Config()

        self.embedding_path = self.config.embedding_path
        self.regime_data_path = self.config.regime_data_path
        self.output_dir = self.config.output_dir
        self.window_size = self.config.window_size
        self.n_clusters = self.config.n_clusters

        os.makedirs(self.output_dir, exist_ok=True)

    def run(self):
        logging.info("🚀 Starting regime clustering pipeline")

        # STEP 1: Load embeddings
        embeddings = np.load(self.embedding_path)
        logging.info(f"📥 Loaded embeddings: shape={embeddings.shape}")

        # STEP 2: Scale embeddings
        scaler = StandardScaler()
        embeddings_scaled = scaler.fit_transform(embeddings)

        # STEP 3: Apply Spectral Clustering
        spectral = SpectralClustering(
            n_clusters=self.n_clusters,
            affinity='nearest_neighbors',
            assign_labels='kmeans',
            random_state=42,
        )
        cluster_labels = spectral.fit_predict(embeddings_scaled)
        logging.info("🔗 Spectral clustering complete.")

        # STEP 4: Dimensionality Reduction
        tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
        tsne_coords = tsne.fit_transform(embeddings_scaled)

        umap = UMAP(n_components=2, random_state=42)
        umap_coords = umap.fit_transform(embeddings_scaled)

        # STEP 5: Load raw data and attach cluster labels
        df = pd.read_csv(self.regime_data_path)
        df['Cluster_ID'] = np.nan
        df.loc[self.window_size - 1 : self.window_size - 1 + len(cluster_labels), 'Cluster_ID'] = cluster_labels
        df['Cluster_ID'] = df['Cluster_ID'].astype('Int64')  # keep nulls for early rows

        # STEP 6: Save merged CSV
        cluster_path = os.path.join(self.output_dir, "cluster_assignments.csv")
        df.to_csv(cluster_path, index=False)
        logging.info(f"💾 Cluster assignments saved: {cluster_path}")

        # STEP 7: Visualizations
        self.plot_scatter(tsne_coords, cluster_labels, "t-SNE", "tsne_plot.png")
        self.plot_scatter(umap_coords, cluster_labels, "UMAP", "umap_plot.png")
        self.plot_timeline(cluster_labels, "timeline.png")
        self.plot_overlay(df, "Close", cluster_labels, "price_overlay.png")

        return df

    def plot_scatter(self, coords, labels, title, filename):
        plt.figure(figsize=(10, 6))
        scatter = plt.scatter(coords[:, 0], coords[:, 1], c=labels, cmap='tab10', edgecolors='k', alpha=0.8)
        plt.title(f"{title} Visualization Colored by Spectral Clustering")
        plt.xlabel(f"{title} Component 1")
        plt.ylabel(f"{title} Component 2")
        plt.colorbar(scatter, label="Cluster ID")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename))
        plt.close()
        logging.info(f"📊 {title} plot saved: {filename}")

    def plot_timeline(self, labels, filename):
        plt.figure(figsize=(14, 4))
        plt.plot(labels, marker='o')
        plt.title("Cluster Regimes Over Time")
        plt.xlabel("Time Index")
        plt.ylabel("Cluster ID")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename))
        plt.close()
        logging.info(f"📈 Timeline plot saved: {filename}")

    def plot_overlay(self, df, price_col, labels, filename):
        df_filtered = df.dropna(subset=['Cluster_ID'])
        plt.figure(figsize=(14, 6))
        plt.plot(df_filtered[price_col].values, label='Price', color='black', alpha=0.7)
        plt.scatter(
            df_filtered.index,
            df_filtered[price_col],
            c=df_filtered['Cluster_ID'],
            cmap='tab10',
            edgecolors='k'
        )
        plt.title("Close Price with Cluster Overlay")
        plt.xlabel("Time Index")
        plt.ylabel("Price")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, filename))
        plt.close()
        logging.info(f"📉 Price overlay saved: {filename}")
