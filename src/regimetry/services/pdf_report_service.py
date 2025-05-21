import os
from pathlib import Path
import re
import textwrap
from fpdf import FPDF
from PIL import Image
from regimetry.config.config import Config
from regimetry.services.analysis_prompt_service import AnalysisPromptService
from regimetry.logger_manager import LoggerManager

logging = LoggerManager.get_logger(__name__)


class PDFReportService:
    """
    Generates a cluster report PDF using fpdf2 with full UTF-8 support and DejaVu font.
    Includes the markdown-style analysis prompt and all cluster visual plots.
    """

    def __init__(self):
        self.config = Config()
        self.experiment_id = self.config.experiment_id
        self.output_dir = self.config.output_dir
        self.pdf_path = os.path.join(self.output_dir, f"{self.experiment_id}_cluster_report.pdf")

        # Font paths
        font_path = Path(self.config.report_font_path)
        bold_font_path = font_path.with_name(font_path.stem + "-Bold" + font_path.suffix)

        logging.info(f"[PDFReportService] font file(s):\n"
                f" → Regular: {font_path}\n"
                f" → Bold:    {bold_font_path}"
)
        if not font_path.exists() or not bold_font_path.exists():
            raise FileNotFoundError(
                f"[PDFReportService] Missing font file(s):\n"
                f" → Regular: {font_path}\n"
                f" → Bold:    {bold_font_path}"
            )

        self.font_path = str(font_path)
        self.bold_font_path = str(bold_font_path)

    def _add_prompt(self, pdf: FPDF, prompt_text: str):
        def clean(text):
            return re.sub(r"[^\u0000-\uFFFF]", "", text)

        pdf.set_font("DejaVu", size=11)
        max_chars = 100
        for i, line in enumerate(prompt_text.split("\n")):
            for subline in textwrap.wrap(clean(line), width=max_chars):
                try:
                    pdf.cell(0, 8, subline, ln=True)
                except Exception as e:
                    logging.error(f"[PDF] 💥 Failed on line {i}: {repr(subline)} → {e}")
                    raise
            pdf.ln(1)


    def _add_image(self, pdf: FPDF, image_path: str, title: str):
        if not os.path.exists(image_path):
            logging.warning(f"[PDFReportService] Image not found: {image_path}")
            return

        pdf.add_page()
        pdf.set_font("DejaVu", style="B", size=12)
        pdf.cell(0, 10, title, ln=True)

        img = Image.open(image_path)
        width, height = img.size
        max_width_mm = 180
        aspect = height / width
        display_height_mm = max_width_mm * aspect

        pdf.image(image_path, x=15, w=max_width_mm, h=display_height_mm)

    def generate_pdf(self):
        logging.info("🧾 Generating PDF cluster report with fpdf2 + UTF-8...")

        prompt_service = AnalysisPromptService()
        prompt_text = prompt_service.get_prompt()

        image_paths = {
            "t-SNE Cluster Overlay": os.path.join(self.output_dir, "tsne_plot.png"),
            "UMAP Cluster Overlay": os.path.join(self.output_dir, "umap_plot.png"),
            "Price Overlay with Clusters": os.path.join(self.output_dir, "price_overlay_plot.png"),
            "Price Overlay with Clusters - Zoomed": os.path.join(self.output_dir, "price_overlay_last150_plot.png"),
            "Cluster Distribution": os.path.join(self.output_dir, "cluster_distribution_plot.png"),
        }

        pdf = FPDF()
        pdf.add_font("DejaVu", "", self.font_path, uni=True)
        pdf.add_font("DejaVu", "B", self.bold_font_path, uni=True)
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()

        pdf.set_font("DejaVu", style="B", size=16)
        pdf.cell(0, 10, f"📊 Cluster Report: {self.experiment_id}", ln=True, align="C")
        pdf.ln(10)

        self._add_prompt(pdf, prompt_text)

        for title, path in image_paths.items():
            self._add_image(pdf, path, title)

        pdf.output(self.pdf_path)
        logging.info(f"✅ PDF saved: {self.pdf_path}")
