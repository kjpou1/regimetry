import numpy as np
import pandas as pd
import json
from scipy.stats import entropy
import seaborn as sns
import matplotlib.pyplot as plt


class RegimeInterpretabilityService:
    def __init__(self, df: pd.DataFrame, cluster_col: str = "Cluster_ID"):
        self.df = df.copy()
        self.cluster_col = cluster_col
        self.cluster_series = df[cluster_col].dropna().astype(int).reset_index(drop=True)
        self.n_clusters = self.cluster_series.nunique()
        self.transition_matrix = None

    def compute_transition_matrix(self):
        matrix = np.zeros((self.n_clusters, self.n_clusters), dtype=int)
        for i in range(len(self.cluster_series) - 1):
            current = self.cluster_series[i]
            next_ = self.cluster_series[i + 1]
            matrix[current, next_] += 1

        row_sums = matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # Prevent division by zero
        self.transition_matrix = matrix / row_sums
        return self.transition_matrix

    def compute_stickiness(self) -> pd.Series:
        if self.transition_matrix is None:
            self.compute_transition_matrix()
        diagonal = np.diag(self.transition_matrix)
        return pd.Series(diagonal, index=[f"Cluster {i}" for i in range(len(diagonal))])

    def compute_entropy(self) -> pd.Series:
        if self.transition_matrix is None:
            self.compute_transition_matrix()
        return pd.Series(
            [entropy(row) for row in self.transition_matrix],
            index=[f"Cluster {i}" for i in range(self.transition_matrix.shape[0])]
        )

    def get_most_likely_transitions(self) -> pd.DataFrame:
        if self.transition_matrix is None:
            self.compute_transition_matrix()
        result = {}
        for i in range(self.transition_matrix.shape[0]):
            row = self.transition_matrix[i].copy()
            row[i] = 0  # Exclude self-loop
            next_idx = np.argmax(row)
            prob = row[next_idx]
            result[f"Cluster {i}"] = (f"Cluster {next_idx}", round(prob, 3)) if prob > 0 else ("None", 0.0)
        return pd.DataFrame.from_dict(result, orient="index", columns=["Next_Cluster", "Probability"])

    def generate_decision_table(self) -> pd.DataFrame:
        stickiness = self.compute_stickiness()
        entropy_vals = self.compute_entropy()
        most_likely_df = self.get_most_likely_transitions()

        table = pd.DataFrame({
            "Stickiness": stickiness,
            "Most_Likely_Next": most_likely_df["Next_Cluster"],
            "Next_Prob": most_likely_df["Probability"],
            "Transition_Entropy": entropy_vals,
        })

        table["Volatile?"] = table["Transition_Entropy"].apply(lambda x: "Yes" if x > 0.4 else "No")

        def strategy_note(row):
            if row["Stickiness"] > 0.90:
                return "✅ Trend-friendly, hold longer"
            elif row["Stickiness"] > 0.75:
                return "🟡 Moderate regime, tighten stop"
            else:
                return "⚠️ Transitional, exit fast"

        table["Strategy_Note"] = table.apply(strategy_note, axis=1)
        return table.round(3)

    def plot_transition_heatmap(self, save_path=None):
        if self.transition_matrix is None:
            self.compute_transition_matrix()

        plt.figure(figsize=(10, 8))
        sns.heatmap(
            self.transition_matrix,
            annot=True,
            fmt=".2f",
            cmap="YlGnBu",
            square=True,
            xticklabels=[f"C{i}" for i in range(self.n_clusters)],
            yticklabels=[f"C{i}" for i in range(self.n_clusters)],
            cbar_kws={"label": "Transition Probability"}
        )
        plt.title("Regime Transition Heatmap")
        plt.xlabel("To Cluster")
        plt.ylabel("From Cluster")
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

    def export_runtime_metadata(self, output_path: str):
        table = self.generate_decision_table()
        metadata = {}

        for cluster_id in range(self.n_clusters):
            key = str(cluster_id)
            row = table.loc[f"Cluster {cluster_id}"]
            metadata[key] = {
                "stickiness": float(row["Stickiness"]),
                "entropy": float(row["Transition_Entropy"]),
                "volatile": row["Volatile?"] == "Yes",
                "note": row["Strategy_Note"]
            }

        with open(output_path, "w") as f:
            json.dump(metadata, f, indent=2)
            