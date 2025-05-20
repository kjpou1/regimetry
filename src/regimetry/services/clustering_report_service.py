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
from regimetry.utils.cluster_utils import normalize_cluster_id, safe_numeric_column

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

    def generate_all(
        self,
        df: pd.DataFrame,
        cluster_labels: np.ndarray,
        tsne_coords: np.ndarray,
        umap_coords: np.ndarray
    ):
        if not self.report_format:
            logging.info("[ClusteringReportService] Skipping report generation due to empty report_format.")
            return

        # === Matplotlib Reports ===
        if 'matplotlib' in self.report_format:
            self.plot_scatter_matplotlib(tsne_coords, cluster_labels, "t-SNE", "tsne_mplot.png")
            self.plot_scatter_matplotlib(umap_coords, cluster_labels, "UMAP", "umap_mplot.png")
            self.plot_timeline_matplotlib(cluster_labels, "timeline_mplot.png")
            self.plot_overlay_matplotlib(df, "Close", "price_overlay_mplot.png")
            self.plot_overlay_matplotlib_last_n(df, "Close", "price_overlay_last150_mplot.png", last_n=150)
            self.plot_cluster_distribution_matplotlib(cluster_labels, filename="cluster_distribution_mplot.png")

        # === Plotly Reports ===
        if 'plotly' in self.report_format:
            self.plot_scatter_plotly(tsne_coords, cluster_labels, "t-SNE", "tsne_plot.html")
            self.plot_scatter_plotly(umap_coords, cluster_labels, "UMAP", "umap_plot.html")
            self.plot_overlay_plotly(df, "Close", "price_overlay_plot.html")
            self.plot_overlay_plotly_last_n(df, "Close", "price_overlay_last150_plot.html", last_n=150)
            self.plot_cluster_distribution_plotly(cluster_labels, filename="cluster_distribution_plot.html")

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
        # Save high-res PNG (requires kaleido)
        png_path = path.replace(".html", ".png")
        fig.write_image(png_path, width=1600, height=900, scale=3)
        logging.info(f"[plotly] Price overlay high-res PNG saved: {png_path}")

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

    def plot_cluster_distribution_matplotlib(self, labels: np.ndarray, filename: str):
        """
        Creates a bar plot of cluster frequency using Matplotlib.
        """
        plt.figure(figsize=(8, 4))
        counts = pd.Series(labels).value_counts().sort_index()
        bars = plt.bar(counts.index.astype(str), counts.values, color=self.palette_colors[:len(counts)])
        plt.title("Cluster Distribution")
        plt.xlabel("Cluster ID")
        plt.ylabel("Number of Windows")
        plt.grid(True, axis='y', linestyle='--', alpha=0.4)
        plt.tight_layout()

        path = os.path.join(self.output_dir, filename)
        plt.savefig(path)
        plt.close()
        logging.info(f"[matplotlib] Cluster distribution saved: {path}")

    def plot_cluster_distribution_plotly(self, labels: np.ndarray, filename: str):
        """
        Creates an interactive bar plot of cluster frequency using Plotly.
        """
        counts = pd.Series(labels).value_counts().sort_index()
        df_plot = pd.DataFrame({"Cluster_ID": counts.index.astype(str), "Count": counts.values})

        fig = px.bar(
            df_plot,
            x="Cluster_ID",
            y="Count",
            title="Cluster Distribution (Interactive)",
            color="Cluster_ID",
            color_discrete_map=self.cluster_color_map,
            category_orders={"Cluster_ID": sorted(df_plot["Cluster_ID"].unique(), key=int, reverse=True)}
        )

        path = os.path.join(self.output_dir, filename)
        fig.write_html(path)
        # Save high-res PNG (requires kaleido)
        png_path = path.replace(".html", ".png")
        fig.write_image(png_path, width=1600, height=900, scale=3)
        logging.info(f"[plotly] Price overlay high-res PNG saved: {png_path}")

        logging.info(f"[plotly] Cluster distribution interactive saved: {path}")

    def plot_overlay_matplotlib(self, df, price_col, filename):
        """
        Plots a Matplotlib price overlay with cluster-colored scatter points.
        Uses df['Cluster_ID'] internally with safe dtype casting.
        """
        df_filtered = df.dropna(subset=["Cluster_ID"]).copy()
        cluster_ids = safe_numeric_column(df_filtered, "Cluster_ID", dtype=int)

        plt.figure(figsize=(14, 6))
        plt.plot(df[price_col].values, label="Price", color="black", alpha=0.6)

        scatter = plt.scatter(
            df_filtered.index,
            df_filtered[price_col],
            c=cluster_ids,
            cmap=self.matplotlib_cmap,
            edgecolors="none",
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
        
    def plot_overlay_plotly(self, df, price_col, filename):
        """
        Plots a Plotly price overlay with cluster-colored scatter markers and rich hover templates.
        Uses df['Cluster_ID'] and dynamically builds hovertemplate per point.
        """
        df_filtered = df.dropna(subset=["Cluster_ID"]).copy()
        df_filtered["Index"] = df_filtered.index

        # Coerce Cluster_ID to Int and back to str for consistent comparison
        df_filtered["Cluster_ID"] = (
            pd.to_numeric(df_filtered["Cluster_ID"], errors="coerce")
            .fillna(-1)
            .astype(int)
            .astype(str)
        )

        # === Hover Template Construction ===
        def format_row(row):
            return (
                f"Time Index: {row.name}<br>"
                f"Cluster: {row['Cluster_ID']}<br><br>"
                f"Price: {row.get(price_col, '—'):.5f}<br>"
                f"ATR: {row.get('ATR', float('nan')):.4f}<br>"
                f"Prediction: {row.get('ML_Trade_Direction', '—')} | Confidence: {row.get('Confidence', 0):.2f} | Signal: {'Strong' if row.get('Confidence', 0) >= 0.80 else 'Weak'}<br>"
                f"ML Confirmed: {'✅' if row.get('ML_Confirmed') else '❌'} | Direction: {row.get('Entry_Side', '-') }<br><br>"
                f"LC_Slope: {row.get('LC_Slope', float('nan')):.5f} | Acceleration: {row.get('LC_Acceleration', float('nan')):.5f}<br>"
                f"Trend: {'Bull' if row.get('Trend_Bull') else 'Bear'} | Expansion: {'✅' if row.get('Expansion_Bull') or row.get('Expansion_Bear') else '❌'} | Divergence: {'✅' if row.get('Momentum_Divergence_Bull') or row.get('Momentum_Divergence_Bear') else '❌'}<br>"
                f"Entry Trigger: {'✅' if row.get('Entry_Trigger') else '❌'} | Entry Confirmed: {'✅' if row.get('Entry_Confirmed') else '❌'}<br><br>"
                f"Stop Loss: {row.get('Stop_Loss_Price', '—')} | TP1: {row.get('Take_Profit_1_Price', '—')} | TP2: {row.get('Take_Profit_2_Price', '—')}<br>"
                f"News Conflict: {'✅' if row.get('News_Conflict') else '❌'}<br><br>"
                f"Trend Agreement: {'✅' if row.get('Trend_Agreement') else '❌'} | Baseline Aligned: {'✅' if row.get('Baseline_Aligned') else '❌'}<br>"
                f"Date: {pd.to_datetime(row.get('Date')).strftime('%a, %b %Y') if pd.notnull(row.get('Date')) else '-'}"
            )

        df_filtered["hovertemplate"] = df_filtered.apply(format_row, axis=1)

        # === Base Price Line ===
        price_trace = go.Scatter(
            x=df.index,
            y=df[price_col],
            mode="lines",
            name="Price",
            line=dict(color="black", width=1),
            opacity=0.5
        )

        # === Cluster Markers ===
        scatter_traces = []
        unique_cluster_ids = sorted(df_filtered["Cluster_ID"].unique(), key=int, reverse=True)
        for cluster_id in unique_cluster_ids:
            cluster_df = df_filtered[df_filtered["Cluster_ID"] == cluster_id]
            logging.info(f"[Cluster Trace] ID: {cluster_id} → {len(cluster_df)} points")
            if cluster_df.empty:
                continue
            scatter_traces.append(go.Scatter(
                x=cluster_df["Index"],
                y=cluster_df[price_col],
                mode="markers",
                name=f"Cluster {cluster_id}",
                marker=dict(
                    color=self.cluster_color_map.get(cluster_id, "gray"),
                    size=6,
                    opacity=0.8
                ),
                customdata=cluster_df["hovertemplate"],
                hovertemplate="%{customdata}"
            ))

        fig = go.Figure(data=[price_trace] + scatter_traces)
        fig.update_layout(
            title="Close Price with Cluster Overlay (Plotly)",
            xaxis_title="Time Index",
            yaxis_title="Price",
            template="plotly_white",
            legend_title="Cluster"
        )

        path = os.path.join(self.output_dir, filename)
        fig.write_html(path)
        # Save high-res PNG (requires kaleido)
        png_path = path.replace(".html", ".png")
        fig.write_image(png_path, width=1600, height=900, scale=3)
        logging.info(f"[plotly] Price overlay high-res PNG saved: {png_path}")

        logging.info(f"[plotly] Price overlay with line and hover saved: {path}")


    def plot_overlay_matplotlib_last_n(self, df, price_col, filename, last_n=150):
        """
        Plots a zoomed-in Matplotlib price overlay using the last N rows.
        Cluster points are color-coded using the current matplotlib colormap.

        Parameters:
        -----------
        df : pd.DataFrame
            Full cluster-labeled DataFrame.
        price_col : str
            Column containing price data (e.g., 'Close').
        filename : str
            Output file name (e.g., 'price_overlay_lastn_mplot.png').
        last_n : int
            Number of most recent time steps to display.
        """
        if len(df) < last_n:
            logging.warning(f"[plot_overlay_matplotlib_last_n] DataFrame only has {len(df)} rows. Using full data.")
            df_filtered = df.copy()
        else:
            df_filtered = df.iloc[-last_n:].copy()

        cluster_ids = safe_numeric_column(df_filtered, "Cluster_ID", dtype=int)

        plt.figure(figsize=(14, 6))
        plt.plot(df_filtered.index, df_filtered[price_col], label="Price", color="black", alpha=0.6)

        scatter = plt.scatter(
            df_filtered.index,
            df_filtered[price_col],
            c=cluster_ids,
            cmap=self.matplotlib_cmap,
            edgecolors="none",
            s=16,
            alpha=0.75
        )

        plt.colorbar(scatter, label="Cluster ID")
        plt.title(f"Last {last_n} Bars – Close Price with Cluster Overlay")
        plt.xlabel("Time Index")
        plt.ylabel("Price")
        plt.grid(True)
        plt.tight_layout()

        path = os.path.join(self.output_dir, filename)
        plt.savefig(path)
        plt.close()
        logging.info(f"[matplotlib] Last {last_n} price overlay saved: {path}")


    def plot_overlay_plotly_last_n(self, df, price_col, filename, last_n=150):
        """
        Plots a zoomed-in Plotly overlay of the last N rows of the clustered data.
        Used to visualize recent regime behavior.

        Parameters:
        -----------
        df : pd.DataFrame
            Full cluster-labeled DataFrame.
        price_col : str
            Name of the price column to plot (e.g., 'Close').
        filename : str
            Output filename (e.g., 'price_overlay_lastn_plot.html').
        last_n : int
            Number of recent rows to include in the plot.
        """
        if len(df) < last_n:
            logging.warning(f"[plot_overlay_plotly_last_n] DataFrame has only {len(df)} rows, using full data.")
            df_filtered = df.copy()
        else:
            df_filtered = df.iloc[-last_n:].copy()

        df_filtered["Index"] = df_filtered.index

        # Coerce Cluster_ID to Int and back to str
        df_filtered["Cluster_ID"] = (
            pd.to_numeric(df_filtered["Cluster_ID"], errors="coerce")
            .fillna(-1)
            .astype(int)
            .astype(str)
        )

        df_filtered["hovertemplate"] = df_filtered.apply(
            lambda row: (
                f"Time Index: {row.name}<br>"
                f"Cluster: {row['Cluster_ID']}<br>"
                f"Price: {row.get(price_col, '—'):.5f}"
            ),
            axis=1
        )

        price_trace = go.Scatter(
            x=df_filtered["Index"],
            y=df_filtered[price_col],
            mode="lines",
            name="Price",
            line=dict(color="black", width=1),
            opacity=0.5
        )

        scatter_traces = []
        unique_cluster_ids = sorted(df_filtered["Cluster_ID"].unique(), key=int, reverse=True)
        for cluster_id in unique_cluster_ids:
            cluster_df = df_filtered[df_filtered["Cluster_ID"] == cluster_id]
            scatter_traces.append(go.Scatter(
                x=cluster_df["Index"],
                y=cluster_df[price_col],
                mode="markers",
                name=f"Cluster {cluster_id}",
                marker=dict(
                    color=self.cluster_color_map.get(cluster_id, "gray"),
                    size=6,
                    opacity=0.85
                ),
                customdata=cluster_df["hovertemplate"],
                hovertemplate="%{customdata}"
            ))

        fig = go.Figure(data=[price_trace] + scatter_traces)
        fig.update_layout(
            title=f"Last {last_n} Bars – Close Price with Cluster Overlay (Plotly)",
            xaxis_title="Time Index",
            yaxis_title="Price",
            template="plotly_white",
            legend_title="Cluster"
        )

        path = os.path.join(self.output_dir, filename)
        fig.write_html(path)
        fig.write_image(path.replace(".html", ".png"), width=1600, height=900, scale=3)
        logging.info(f"[plotly] Zoomed-in overlay plot saved: {path}")
