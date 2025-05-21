from regimetry.config.config import Config


class AnalysisPromptService:
    """
    Generates standardized cluster analysis prompts using the active Config values.
    Emphasizes using the zoomed last-150-bars overlay for recent regime analysis.
    """

    def __init__(self):
        self.config = Config()
        self.instrument = self.config.instrument or "Unknown"
        self.experiment_id = self.config.experiment_id
        self.config_file = self.config.config_path

    def get_prompt(self) -> str:
        """
        Returns a formatted Markdown-style cluster analysis prompt for use in reports or reviews.
        Includes note to use zoomed last-150-bar view for final regime commentary.
        """
        if self.config.encoding_method.lower().startswith("sin"):
            encoding_style_line = f"**Encoding Style**: {self.config.encoding_style.capitalize()}  \n"
        else:
            encoding_style_line = "*Note: Encoding style is only applicable to sinusoidal encodings.*\n"

        return f"""
📊 Cluster Report Analysis Prompt  
You are an expert in time-series regime detection using transformer embeddings, spectral clustering, and price overlays. Analyze the clustering report for the instrument below.

**Instrument**: {self.instrument}  
**Experiment ID**: {self.experiment_id}  
**Config File**: {self.config_file}  
**Window Size**: {self.config.window_size}  
**Embedding Dim**: {self.config.embedding_dim}  
**Encoding Method**: {self.config.encoding_method.capitalize()}  
{encoding_style_line}**Number of Clusters**: {self.config.n_clusters}  

You are provided with:
- A t-SNE plot with cluster ID overlay  
- A UMAP plot with cluster ID overlay  
- A price overlay with time-aligned cluster segments (full length)  
- A **zoomed price overlay showing the last 150 bars**  
- A cluster distribution histogram  

#### Your tasks:
1. **Regime Separation**: Are the clusters well-separated in t-SNE and UMAP? Are there overlaps or tight transitions?
2. **Temporal Alignment**: Do the cluster transitions correspond to meaningful price trend changes in the price overlay?
3. **Final Regime**: *Using the zoomed 150-bar chart*, determine which cluster governs the final price regime. Is it stable or volatile? Large or small?
4. **Cluster Sizes**: Are any clusters over/under-represented in the distribution?
5. **Dimensional Suitability**: Does the embedding dimension ({self.config.embedding_dim}) appear sufficient compared to a higher-dim baseline?
6. **Improvement Areas**: Suggest if the current number of clusters (`nc={self.config.n_clusters}`) is adequate or whether a higher/lower `n_clusters` may capture better structure.
7. **Final Verdict**: Would you approve this config for production regime labeling?

> 📌 For recent regime analysis (Tasks 2–3), focus on the **zoomed last 150 bars** plot for higher accuracy. Use the full-length overlay only for global context.

Please return a concise but structured summary with bullets and decision recommendations.
""".strip()

    def get_metadata(self) -> dict:
        """
        Returns metadata for use in filenames, PDFs, dashboards, or logging.
        """
        return {
            "instrument": self.instrument,
            "experiment_id": self.experiment_id,
            "config_path": self.config_file,
            "window_size": self.config.window_size,
            "embedding_dim": self.config.embedding_dim,
            "encoding_method": self.config.encoding_method,
            "encoding_style": self.config.encoding_style,
            "n_clusters": self.config.n_clusters
        }
