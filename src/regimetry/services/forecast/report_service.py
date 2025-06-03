import json
import os
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from regimetry.config import Config
from regimetry.services.forecast.evaluation_service import ForecastEvaluationService


class ForecastReportService:
    def __init__(
        self,
        evaluator: Optional[ForecastEvaluationService] = None,
        history_path: Optional[str] = None,
        output_dir: Optional[str] = None,
    ):
        """
        Generate visual reports from forecast evaluation results.

        Args:
            evaluator: Optional ForecastEvaluationService instance.
            history_path: Optional path to history.json (loss curve).
            output_dir: Override output directory. Defaults to Config().output_dir.
        """
        self.config = Config()
        self.output_dir = output_dir or self.config.output_dir
        self.evaluator = evaluator
        self.history_path = history_path or os.path.join(
            self.output_dir, "forecast_history.json"
        )

        self.metrics = evaluator.get_metrics() if evaluator else self._load_metrics()
        self.confusion_matrix = self.metrics["confusion_matrix"]
        self.f1_scores = {
            str(row["Cluster"]): row["F1"]
            for row in self.metrics.get("performance", [])
        }

    @classmethod
    def from_evaluator(
        cls,
        evaluator: ForecastEvaluationService,
        output_dir: Optional[str] = None,
        history_path: Optional[str] = None,
    ) -> "ForecastReportService":
        return cls(
            evaluator=evaluator, output_dir=output_dir, history_path=history_path
        )

    @classmethod
    def from_files(
        cls,
        output_dir: Optional[str] = None,
        history_path: Optional[str] = None,
    ) -> "ForecastReportService":
        return cls(evaluator=None, output_dir=output_dir, history_path=history_path)

    def _load_metrics(self):
        path = os.path.join(self.output_dir, "evaluation_metrics.json")
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _format_confusion_matrix_table(self) -> str:
        """
        Format the confusion matrix as a monospaced markdown table with aligned P/T labels.
        """
        cm = np.array(self.confusion_matrix)
        n = cm.shape[0]

        col_width = 5  # Enough for 4-digit numbers and padding

        # Header row with P# labels
        header = (
            " " * 8
            + "|"
            + "".join([f"{f'P{i}':>{col_width}}" for i in range(n)])
            + "\n"
        )
        divider = "-" * 8 + "+" + "-" * (col_width * n) + "\n"

        # Rows with T# labels
        rows = ""
        for i, row in enumerate(cm):
            rows += (
                f"{f'T{i}':<8}|"
                + "".join([f"{int(val):>{col_width}}" for val in row])
                + "\n"
            )

        legend = (
            "\n**Legend**\n"
            "- `T#`: True cluster ID  \n"
            "- `P#`: Predicted cluster ID\n"
        )

        return f"```\n{header}{divider}{rows}```\n{legend}"

    def generate_loss_curve(self):
        """Generate and save training loss curve plot from history.json."""
        if not os.path.exists(self.history_path):
            print(f"[WARN] No history.json found at {self.history_path}")
            return

        with open(self.history_path, "r", encoding="utf-8") as f:
            history = json.load(f)

        loss = history.get("loss", [])
        val_loss = history.get("val_loss", [])

        plt.figure(figsize=(8, 4))
        plt.plot(loss, label="Train Loss")
        if val_loss:
            plt.plot(val_loss, label="Val Loss")
        plt.title("Loss Curve")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "loss_curve.png"))
        plt.close()

    def generate_confusion_matrix_plot(self):
        """Generate and save a heatmap of the confusion matrix."""
        plt.figure(figsize=(8, 6))
        sns.heatmap(self.confusion_matrix, annot=True, fmt=".0f", cmap="Blues")
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted Cluster")
        plt.ylabel("True Cluster")
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "confusion_matrix.png"))
        plt.close()

    def generate_per_cluster_f1_plot(self):
        """Generate and save a bar chart of per-cluster F1 scores."""
        if not self.f1_scores:
            print("[WARN] No per-cluster F1 scores found.")
            return

        cluster_ids = list(map(int, self.f1_scores.keys()))
        scores = [self.f1_scores[str(cid)] for cid in cluster_ids]

        plt.figure(figsize=(8, 4))
        sns.barplot(x=scores, y=cluster_ids, orient="h")
        plt.title("Per-Cluster F1 Scores")
        plt.xlabel("F1 Score")
        plt.ylabel("Cluster ID")
        plt.xlim(0, 1)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "per_cluster_f1.png"))
        plt.close()

    def generate_normalized_confusion_matrix_plot(self):
        """Generate and save a normalized confusion matrix heatmap."""
        cm = np.array(self.confusion_matrix, dtype=np.float32)
        row_sums = cm.sum(axis=1, keepdims=True)
        normalized_cm = np.divide(cm, row_sums, where=row_sums != 0)

        plt.figure(figsize=(8, 6))
        sns.heatmap(normalized_cm, annot=True, fmt=".2f", cmap="Blues", cbar=True)
        plt.title("Normalized Confusion Matrix")
        plt.xlabel("Predicted Cluster")
        plt.ylabel("True Cluster")
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "confusion_matrix_normalized.png"))
        plt.close()

    def generate_cluster_sample_count_plot(self):
        distribution = self.metrics.get("cluster_distribution", [])

        if not distribution:
            print("[WARN] No cluster_distribution found for count plot.")
            return

        clusters = [d["Cluster"] for d in distribution]
        counts = [d["Total Count"] for d in distribution]

        plt.figure(figsize=(8, 4))
        sns.barplot(x=counts, y=clusters, orient="h")
        plt.title("Per-Cluster Sample Counts")
        plt.xlabel("Sample Count")
        plt.ylabel("Cluster ID")
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "cluster_sample_counts.png"))
        plt.close()

    def generate_markdown_report(self):
        report_path = os.path.join(self.output_dir, "evaluation_report.md")

        performance = self.metrics.get("performance", [])
        cluster_distribution = self.metrics.get("cluster_distribution", [])
        top = sorted(performance, key=lambda x: x["F1"], reverse=True)[:3]
        bottom = [c for c in sorted(performance, key=lambda x: x["F1"]) if c["F1"] > 0][
            :3
        ]

        with open(report_path, "w", encoding="utf-8") as f:
            f.write("# 📈 Forecast Evaluation Report\n\n")

            f.write("## 🔝 Top 3 Clusters by F1 Score\n")
            for c in top:
                f.write(
                    f"- Cluster {c['Cluster']}: F1 = {c['F1']:.3f} (Precision = {c['Precision']:.3f}, Recall = {c['Recall']:.3f})\n"
                )

            f.write("\n## ❌ Bottom 3 Clusters by F1 Score (non-zero only)\n")
            for c in bottom:
                f.write(
                    f"- Cluster {c['Cluster']}: F1 = {c['F1']:.3f} (Precision = {c['Precision']:.3f}, Recall = {c['Recall']:.3f})\n"
                )

            f.write("\n## 📊 Per-Cluster Sample Counts\n")
            f.write("| Cluster | Train Count | Val Count | Total Count |\n")
            f.write("|---------|--------------|-----------|--------------|\n")
            for row in cluster_distribution:
                f.write(
                    f"| {row['Cluster']} | {row['Train Count']} | {row['Val Count']} | {row['Total Count']} |\n"
                )

            # Confusion Matrix Markdown Table
            f.write("\n## 🔍 Confusion Matrix Table (Train Only)\n")
            f.write(self._format_confusion_matrix_table())

            f.write("\n\n## 🖼️ Plots\n")
            f.write("![Loss Curve](loss_curve.png)\n")
            f.write("![Confusion Matrix](confusion_matrix.png)\n")
            f.write("![Normalized Confusion Matrix](confusion_matrix_normalized.png)\n")
            f.write("![Per-Cluster F1](per_cluster_f1.png)\n")
            f.write("![Cluster Sample Counts](cluster_sample_counts.png)\n")
            f.write("![Transition Misfire Matrix](transition_misfire_matrix.png)\n")

        print(f"📄 Markdown report written to: {report_path}")

    def generate_transition_misfire_matrix_plot(self):
        if not self.evaluator or not hasattr(
            self.evaluator, "true_next_cluster_labels"
        ):
            print("[WARN] Transition map requires evaluator with transition labels.")
            return

        true = np.array(self.evaluator.true_next_cluster_labels)
        pred = np.array(self.evaluator.predicted_next_cluster_labels)

        if true.shape != pred.shape:
            print("[ERROR] Shape mismatch in transition label arrays.")
            return

        n_clusters = int(max(true.max(), pred.max()) + 1)
        matrix = np.zeros((n_clusters, n_clusters), dtype=int)

        for t, p in zip(true, pred):
            matrix[int(t), int(p)] += 1

        norm_matrix = matrix / matrix.sum(axis=1, keepdims=True)
        norm_matrix = np.nan_to_num(norm_matrix)

        plt.figure(figsize=(8, 6))
        sns.heatmap(norm_matrix, annot=True, fmt=".2f", cmap="Oranges", cbar=True)
        plt.title("Transition Misfire Map\n(True Cluster[t+1] vs Predicted)")
        plt.xlabel("Predicted Cluster")
        plt.ylabel("True Next Cluster")
        plt.tight_layout()
        out_path = os.path.join(self.output_dir, "transition_misfire_matrix.png")
        plt.savefig(out_path)
        plt.close()
        print(f"📉 Transition misfire map saved: {out_path}")

    def generate_all(self):
        """Run all report generators."""
        self.generate_loss_curve()
        self.generate_confusion_matrix_plot()
        self.generate_normalized_confusion_matrix_plot()
        self.generate_per_cluster_f1_plot()
        self.generate_cluster_sample_count_plot()
        self.generate_transition_misfire_matrix_plot()
        self.generate_markdown_report()
