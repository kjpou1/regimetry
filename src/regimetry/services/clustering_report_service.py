# src/regimetry/services/clustering_report_service.py

import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from matplotlib.colors import ListedColormap

from regimetry.config.config import Config
from regimetry.logger_manager import LoggerManager

logging = LoggerManager.get_logger(__name__)


class ClusteringReportService:
    """
    Generates clustering visualizations for regime detection, including t-SNE, UMAP, timeline,
    and Close Price overlays. Supports both Matplotlib and Plotly backends.

    Uses consistent coloring from seaborn palettes, controlled by `report_palette` in config.
    """

    def __init__(self):
        self.config = Config()
        self.output_dir = self.config.output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        self.report_format = self.config.report_format or []
        self.palette_name = getattr(self.config, "report_palette", "tab10")
        self.n_clusters = getattr(self.config, "n_clusters", 8)

        # Build consistent color mappings
        try:
            palette = sns.color_palette(self.palette_name, n_colors=self.n_clusters).as_hex()
            self.cluster_color_map = {str(i): color for i, color in enumerate(palette)}
            self.palette_colors = list(self.cluster_color_map.values())
            self.matplotlib_cmap = ListedColormap(self.palette_colors)
        except Exception:
            logging.warning(f"[ClusteringReportService] Invalid palette '{self.palette_name}', defaulting to 'tab10'")
            fallback = sns.color_palette("tab10", n_colors=self.n_clusters).as_hex()
            self.cluster_color_map = {str(i): color for i, color in enumerate(fallback)}
            self.palette_colors = list(self.cluster_color_map.values())
            self.matplotlib_cmap = ListedColormap(self.palette_colors)

    def generate_all(self, df: pd.DataFrame, cluster_labels: np.ndarray, tsne_coords: np.ndarray, umap_coords: np.ndarray):
        if not self.report_format:
            logging.info("[ClusteringReportService] Skipping report generation due to empty report_format.")
            return

        if 'matplotlib' in self.report_format:
            self.plot_scatter_matplotlib(tsne_coords, cluster_labels, "t-SNE", "tsne_plot.png")
            self.plot_scatter_matplotlib(umap_coords, cluster_labels, "UMAP", "umap_plot.png")
            self.plot_timeline_matplotlib(cluster_labels, "timeline.png")
            self.plot_overlay_matplotlib(df, "Close", cluster_labels, "price_overlay.png")

        if 'plotly' in self.report_format:
            self.plot_scatter_plotly(tsne_coords, cluster_labels, "t-SNE", "tsne_plot.html")
            self.plot_scatter_plotly(umap_coords, cluster_labels, "UMAP", "umap_plot.html")
            self.plot_overlay_plotly(df, "Close", cluster_labels, "price_overlay.html")

    def plot_scatter_matplotlib(self, coords, labels, title, filename):
        plt.figure(figsize=(10, 6))
        scatter = plt.scatter(coords[:, 0], coords[:, 1], c=labels, cmap=self.matplotlib_cmap, edgecolors='none', alpha=0.8)
        plt.title(f"{title} Visualization")
        plt.xlabel(f"{title} Component 1")
        plt.ylabel(f"{title} Component 2")
        plt.colorbar(scatter, label="Cluster ID")
        plt.grid(True)
        plt.tight_layout()
        path = os.path.join(self.output_dir, filename)
        plt.savefig(path)
        plt.close()
        logging.info(f"[matplotlib] {title} plot saved: {path}")

    def plot_scatter_plotly(self, coords, labels, title, filename):
        df_plot = pd.DataFrame(coords, columns=['x', 'y'])
        df_plot['Cluster_ID'] = labels.astype(str)
        unique_cluster_ids = sorted(df_plot['Cluster_ID'].unique(), key=int, reverse=True)
        category_orders = {"Cluster_ID": unique_cluster_ids}

        fig = px.scatter(
            df_plot,
            x='x',
            y='y',
            color='Cluster_ID',
            title=f"{title} Visualization (Plotly)",
            color_discrete_map=self.cluster_color_map,
            category_orders=category_orders
        )
        path = os.path.join(self.output_dir, filename)
        fig.write_html(path)
        logging.info(f"[plotly] {title} interactive plot saved: {path}")

    def plot_timeline_matplotlib(self, labels, filename):
        plt.figure(figsize=(14, 4))
        plt.plot(labels, marker='o')
        plt.title("Cluster Regimes Over Time")
        plt.xlabel("Time Index")
        plt.ylabel("Cluster ID")
        plt.grid(True)
        plt.tight_layout()
        path = os.path.join(self.output_dir, filename)
        plt.savefig(path)
        plt.close()
        logging.info(f"[matplotlib] Timeline plot saved: {path}")

    def plot_overlay_matplotlib(self, df, price_col, labels, filename):
        df_filtered = df.dropna(subset=['Cluster_ID'])
        plt.figure(figsize=(14, 6))
        plt.plot(df_filtered[price_col].values, label='Price', color='black', alpha=0.7)

        scatter = plt.scatter(
            df_filtered.index,
            df_filtered[price_col],
            c=df_filtered['Cluster_ID'],
            cmap=self.matplotlib_cmap,
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
        path = os.path.join(self.output_dir, filename)
        plt.savefig(path)
        plt.close()
        logging.info(f"[matplotlib] Price overlay saved: {path}")

    def plot_overlay_plotly(self, df, price_col, labels, filename):
        df_filtered = df.dropna(subset=['Cluster_ID']).copy()
        df_filtered['Index'] = df_filtered.index
        df_filtered['Cluster_ID'] = df_filtered['Cluster_ID'].astype(str)

        price_trace = go.Scatter(
            x=df_filtered['Index'],
            y=df_filtered[price_col],
            mode='lines',
            name='Price',
            line=dict(color='black', width=1),
            opacity=0.6
        )

        unique_cluster_ids = sorted(df_filtered['Cluster_ID'].unique(), key=int, reverse=True)
        category_orders = {"Cluster_ID": unique_cluster_ids}

        scatter_fig = px.scatter(
            df_filtered,
            x='Index',
            y=price_col,
            color='Cluster_ID',
            labels={'Index': 'Time Index', price_col: 'Close'},
            color_discrete_map=self.cluster_color_map,
            category_orders=category_orders
        )

        fig = go.Figure(data=[price_trace] + list(scatter_fig.data))
        fig.update_layout(
            title="Close Price with Cluster Overlay (Plotly)",
            xaxis_title="Time Index",
            yaxis_title="Price",
            template="plotly_white",
            legend_title="Legend"
        )

        path = os.path.join(self.output_dir, filename)
        fig.write_html(path)
        logging.info(f"[plotly] Price overlay with line saved: {path}")
