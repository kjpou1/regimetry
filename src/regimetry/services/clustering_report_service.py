# src/regimetry/services/clustering_report_service.py

import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from regimetry.logger_manager import LoggerManager

logging = LoggerManager.get_logger(__name__)


class ClusteringReportService:
    """
    Service class responsible for generating visual reports for clustering results.

    This includes:
    - Scatter plots for t-SNE and UMAP projections
    - Timeline plot of cluster transitions over time
    - Overlay of cluster regimes on price series
    """

    def __init__(self, output_dir: str):
        """
        Initialize the report service with the output directory.

        Args:
            output_dir (str): Directory where plots will be saved.
        """
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def generate_all(self, df: pd.DataFrame, cluster_labels: np.ndarray, tsne_coords: np.ndarray, umap_coords: np.ndarray):
        """
        Generate all available reports: t-SNE, UMAP, timeline, and price overlay.

        Args:
            df (pd.DataFrame): Input dataframe containing 'Close' prices and 'Cluster_ID'.
            cluster_labels (np.ndarray): Array of cluster labels.
            tsne_coords (np.ndarray): 2D t-SNE embedding coordinates.
            umap_coords (np.ndarray): 2D UMAP embedding coordinates.
        """
        self.plot_scatter(tsne_coords, cluster_labels, "t-SNE", "tsne_plot.png")
        self.plot_scatter(umap_coords, cluster_labels, "UMAP", "umap_plot.png")
        self.plot_timeline(cluster_labels, "timeline.png")
        self.plot_overlay(df, "Close", cluster_labels, "price_overlay.png")

    def plot_scatter(self, coords, labels, title, filename):
        """
        Create a scatter plot using reduced 2D coordinates (e.g. t-SNE or UMAP).

        Args:
            coords (np.ndarray): 2D coordinates for each point.
            labels (np.ndarray): Cluster ID for each point.
            title (str): Plot title and axis label prefix.
            filename (str): Filename to save the plot.
        """
        plt.figure(figsize=(10, 6))
        scatter = plt.scatter(
            coords[:, 0], coords[:, 1],
            c=labels, cmap='tab10',
            edgecolors='none', alpha=0.8
        )
        plt.title(f"{title} Visualization Colored by Spectral Clustering")
        plt.xlabel(f"{title} Component 1")
        plt.ylabel(f"{title} Component 2")
        plt.colorbar(scatter, label="Cluster ID")
        plt.grid(True)
        plt.tight_layout()

        output_path = os.path.join(self.output_dir, filename)
        plt.savefig(output_path)
        plt.close()
        logging.info(f"📊 {title} plot saved: {output_path}")

    def plot_timeline(self, labels, filename):
        """
        Plot cluster regime transitions over time as a sequence.

        Args:
            labels (np.ndarray): Cluster labels ordered by time.
            filename (str): Filename to save the plot.
        """
        plt.figure(figsize=(14, 4))
        plt.plot(labels, marker='o')
        plt.title("Cluster Regimes Over Time")
        plt.xlabel("Time Index")
        plt.ylabel("Cluster ID")
        plt.grid(True)
        plt.tight_layout()

        output_path = os.path.join(self.output_dir, filename)
        plt.savefig(output_path)
        plt.close()
        logging.info(f"📈 Timeline plot saved: {output_path}")

    def plot_overlay(self, df, price_col, labels, filename):
        """
        Overlay cluster regime coloring on top of the price chart.

        Args:
            df (pd.DataFrame): Input dataframe with 'Cluster_ID' and price column.
            price_col (str): Column name for the price series to overlay (e.g. 'Close').
            labels (np.ndarray): Cluster labels (not used here, kept for consistency).
            filename (str): Filename to save the overlay plot.
        """
        df_filtered = df.dropna(subset=['Cluster_ID'])

        plt.figure(figsize=(14, 6))
        plt.plot(df_filtered[price_col].values, label='Price', color='black', alpha=0.7)

        scatter = plt.scatter(
            df_filtered.index,
            df_filtered[price_col],
            c=df_filtered['Cluster_ID'],
            cmap='tab10',
            edgecolors='none',
            s=16,
            alpha=0.75
        )

        plt.colorbar(scatter, label="Cluster ID")
        plt.title("Close Price with Cluster Overlay")
        plt.xlabel("Time Index")
        plt.ylabel("Price")
        plt.grid(True)
        plt.tight_layout()

        output_path = os.path.join(self.output_dir, filename)
        plt.savefig(output_path)
        plt.close()
        logging.info(f"📉 Price overlay saved: {output_path}")
