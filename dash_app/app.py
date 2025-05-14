import os
import dash
from dash import html, dcc, Input, Output, State
import dash_bootstrap_components as dbc
import base64
import tempfile
import yaml
import seaborn as sns

from regimetry.config.config import Config

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "🧠 Regime Clustering Dashboard"

# === Load default YAML config on app startup ===
DEFAULT_CONFIG_PATH = "./configs/full_config.yaml"
Config().load_from_yaml(DEFAULT_CONFIG_PATH)
default_report_dir = Config().output_dir

# === Helper to read Plotly HTML files ===
def load_plotly_html(report_dir, filename):
    if not report_dir:
        return "<p>No report directory available.</p>"
    path = os.path.join(report_dir, filename)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    return f"<p>Plot not found: {filename}</p>"

# === Helper to render palette block ===
def render_palette_preview(palette_name, n_clusters):
    try:
        colors = sns.color_palette(palette_name, n_colors=n_clusters).as_hex()
    except Exception:
        colors = sns.color_palette("tab10", n_colors=n_clusters).as_hex()

    return html.Div([
        html.H5("🎨 Cluster Color Palette", className="mt-4 mb-3"),
        html.Div([
            html.Div([
                html.Div(style={
                    "backgroundColor": color,
                    "width": "40px",
                    "height": "40px",
                    "display": "inline-block",
                    "marginRight": "10px",
                    "border": "1px solid #ccc"
                }),
                html.Span(f"Cluster {i}", style={"verticalAlign": "middle"})
            ], className="mb-2")
            for i, color in enumerate(colors)
        ])
    ], className="ms-3 mt-2")

# === Layout ===
app.layout = dbc.Container([
    dcc.Store(id="report-dir-store", data=default_report_dir),
    
    html.H2("🧠 Regime Clustering Dashboard", className="my-4 text-center"),

    dbc.Row([
        dbc.Col([
            dcc.Upload(
                id='yaml-upload',
                children=html.Div([
                    "📁 Drag & Drop or ",
                    html.B("Select a YAML Config File")
                ]),
                style={
                    'width': '100%',
                    'height': '100px',
                    'lineHeight': '100px',
                    'borderWidth': '1px',
                    'borderStyle': 'dashed',
                    'borderRadius': '5px',
                    'textAlign': 'center',
                    'marginBottom': '20px'
                },
                multiple=False
            )
        ])
    ]),

    dbc.Tabs(id="main-tabs", active_tab="price", children=[
        dbc.Tab(label="📉 Price Overlay", tab_id="price"),
        dbc.Tab(label="🌀 t-SNE", tab_id="tsne"),
        dbc.Tab(label="🔮 UMAP", tab_id="umap"),
        dbc.Tab(label="🎨 Palette Preview", tab_id="palette")
    ]),
    
    html.Div(id="tab-content")
], fluid=True)


# === Tab Content Callback ===
@app.callback(
    Output("tab-content", "children"),
    Input("main-tabs", "active_tab"),
    State("report-dir-store", "data")
)
def render_tab(tab, report_dir):
    if not report_dir:
        return html.Div("⚠️ No report directory loaded.")

    if tab == "price":
        return html.Iframe(
            srcDoc=load_plotly_html(report_dir, "price_overlay.html"),
            style={"width": "100%", "height": "700px", "border": "none"}
        )
    elif tab == "tsne":
        return html.Iframe(
            srcDoc=load_plotly_html(report_dir, "tsne_plot.html"),
            style={"width": "100%", "height": "600px", "border": "none"}
        )
    elif tab == "umap":
        return html.Iframe(
            srcDoc=load_plotly_html(report_dir, "umap_plot.html"),
            style={"width": "100%", "height": "600px", "border": "none"}
        )
    elif tab == "palette":
        cfg = Config()
        return render_palette_preview(cfg.report_palette, cfg.n_clusters)

    return html.Div("🔍 Select a tab.")


# === YAML Upload Callback ===
@app.callback(
    Output("report-dir-store", "data"),
    Input("yaml-upload", "contents"),
    State("yaml-upload", "filename"),
    prevent_initial_call=True
)
def update_config_from_upload(contents, filename):
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)

    with tempfile.NamedTemporaryFile(mode="w+b", suffix=".yaml", delete=False) as tmp:
        tmp.write(decoded)
        tmp_path = tmp.name

    Config().load_from_yaml(tmp_path)
    return Config().output_dir


if __name__ == "__main__":
    app.run(debug=True)
