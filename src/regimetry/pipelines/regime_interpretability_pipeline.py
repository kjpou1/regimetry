import os
import pandas as pd

from regimetry.services.regime_interpretability_service import RegimeInterpretabilityService


def run(
    input_path: str,
    output_dir: str,
    cluster_col: str = "Cluster_ID",
    save_csv: bool = True,
    save_heatmap: bool = True,
    save_json: bool = False
):
    """
    Entry point for the regime interpretability pipeline.

    Parameters:
    - input_path: path to the cluster-labeled CSV file.
    - output_dir: where to write all outputs (heatmap, CSVs).
    - cluster_col: column name used for regime IDs (default: Cluster_ID).
    - save_csv: whether to save decision table + transition matrix to CSV.
    - save_heatmap: whether to save the transition matrix heatmap as PNG.
    """

    # Load cluster-labeled signal dataset
    df = pd.read_csv(input_path)

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Initialize interpretability service
    service = RegimeInterpretabilityService(df, cluster_col=cluster_col)

    # Compute all structure metadata
    service.compute_transition_matrix()
    decision_table = service.generate_decision_table()

    # Save heatmap (optional)
    if save_heatmap:
        heatmap_path = os.path.join(output_dir, "transition_matrix_heatmap.png")
        service.plot_transition_heatmap(save_path=heatmap_path)
        print(f"✅ Saved heatmap: {heatmap_path}")

    # Save CSV outputs (optional)
    if save_csv:
        decision_table_path = os.path.join(output_dir, "regime_decision_table.csv")
        matrix_csv_path = os.path.join(output_dir, "transition_matrix.csv")

        decision_table.to_csv(decision_table_path, index=True)
        pd.DataFrame(service.transition_matrix).to_csv(matrix_csv_path, index=False)

        print(f"✅ Saved decision table: {decision_table_path}")
        print(f"✅ Saved transition matrix: {matrix_csv_path}")

    if save_json:
        json_path = os.path.join(output_dir, "regime_metadata.json")
        service.export_runtime_metadata(json_path)
        print(f"✅ Saved runtime metadata JSON: {json_path}")
