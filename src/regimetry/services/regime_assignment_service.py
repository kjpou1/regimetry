import json
import os

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import confusion_matrix

from regimetry.config.config import Config
from regimetry.logger_manager import LoggerManager
from regimetry.utils.cluster_utils import (
    attach_cluster_labels,
    verify_cluster_alignment,
)

logger = LoggerManager.get_logger(__name__)


class RegimeAssignmentService:
    """
    Handles regime assignment with stable cluster label alignment across multiple runs.
    Ensures regime continuity by using Hungarian alignment against a stored baseline.
    """

    def __init__(self):
        self.config = Config()
        self.output_dir = self.config.output_dir
        self.window_size = self.config.window_size
        self.n_clusters = self.config.n_clusters
        self.baseline_dir = self.config.baseline_metadata_dir

        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.baseline_dir, exist_ok=True)

    def assign_and_align(
        self, regime_df: pd.DataFrame, cluster_labels: np.ndarray
    ) -> pd.DataFrame:
        """
        Attach raw cluster labels to the regime dataframe, then remap them using
        a fixed baseline alignment strategy if applicable.

        Returns:
            regime_df: DataFrame with final aligned cluster assignments.
        """
        logger.info("🧱 Attaching cluster labels to regime data...")
        regime_df = attach_cluster_labels(regime_df, cluster_labels)
        verify_cluster_alignment(regime_df, self.window_size)

        # Save pre-mapped regime file for inspection/debugging
        # prefinal_path = os.path.join(self.output_dir, "regime_assignments_pre.csv")
        # regime_df.to_csv(prefinal_path, index=False, encoding="utf-8")

        baseline_path = os.path.join(
            self.baseline_dir, "regime_assignments_baseline.csv"
        )
        mapping_path = os.path.join(self.baseline_dir, "regime_label_mapping.json")

        if os.path.exists(baseline_path):
            logger.info("🔁 Found baseline. Attempting cluster label alignment...")

            # Generate a new alignment mapping from current to baseline
            mapping = self._align_cluster_labels(regime_df, baseline_path)

            if mapping and not os.path.exists(mapping_path):
                with open(mapping_path, "w", encoding="utf-8") as f:
                    json.dump(mapping, f, indent=2)
                logger.info(f"💾 Saved alignment mapping: {mapping_path}")
            else:
                logger.info("✅ Existing mapping already applied or unchanged.")

            # Apply mapping to align labels
            regime_df["Cluster_ID"] = regime_df["Cluster_ID"].apply(
                lambda x: mapping.get(int(x), x) if pd.notna(x) else x
            )

        else:
            # First-time run — create baseline, but do not generate mapping
            logger.info("📌 No baseline found. Saving current labels as baseline...")
            current_valid = regime_df["Cluster_ID"].dropna().astype(int)
            current_valid.to_frame("Cluster_ID").to_csv(
                baseline_path, index=False, encoding="utf-8"
            )

        # Save final aligned cluster assignments
        final_path = os.path.join(self.output_dir, "regime_assignments.csv")
        regime_df.to_csv(final_path, index=False, encoding="utf-8")
        logger.info(f"✅ Regime assignments saved: {final_path}")

        return regime_df

    def _align_cluster_labels(
        self, regime_df: pd.DataFrame, baseline_path: str
    ) -> dict:
        """
        Align current cluster labels to a baseline using the Hungarian algorithm.

        Args:
            regime_df: DataFrame containing current cluster labels in 'Cluster_ID'.
            baseline_path: Path to CSV file with baseline 'Cluster_ID' column.

        Returns:
            mapping (dict): A mapping from current cluster IDs → baseline cluster IDs.
        """
        baseline_series = pd.read_csv(baseline_path)["Cluster_ID"].dropna().astype(int)
        current_series = regime_df["Cluster_ID"].dropna().astype(int)

        min_len = min(len(current_series), len(baseline_series))
        current_trimmed = current_series.iloc[:min_len].to_numpy()
        baseline_trimmed = baseline_series.iloc[:min_len].to_numpy()

        # Build confusion matrix and compute optimal assignment
        C = confusion_matrix(
            baseline_trimmed, current_trimmed, labels=range(self.n_clusters)
        )
        row_ind, col_ind = linear_sum_assignment(-C)

        mapping = {int(col): int(row) for row, col in zip(row_ind, col_ind)}

        logger.info(f"📐 Alignment mapping calculated: {mapping}")
        return mapping
