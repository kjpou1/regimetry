# dash_app.py

import os
import dash
from dash import html, dcc
import plotly.io as pio

REPORT_DIR = "./artifacts/reports/EUR_USD"  # Adjust as needed

def load_plotly_html(filename):
    path = os.path.join(REPORT_DIR, filename)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    return "<p>Plot not found: {}</p>".format(filename)

app = dash.Dash(__name__)
app.title = "Regime Clustering Dashboard"

app.layout = html.Div([
    html.H1("🧠 Regime Clustering Dashboard", style={"textAlign": "center"}),
    
    dcc.Tabs([
        dcc.Tab(label='📉 Price Overlay', children=[
            html.Iframe(
                srcDoc=load_plotly_html("price_overlay.html"),
                style={"width": "100%", "height": "700px", "border": "none"}
            )
        ]),
        dcc.Tab(label='🌀 t-SNE', children=[
            html.Iframe(
                srcDoc=load_plotly_html("tsne_plot.html"),
                style={"width": "100%", "height": "600px", "border": "none"}
            )
        ]),
        dcc.Tab(label='🔮 UMAP', children=[
            html.Iframe(
                srcDoc=load_plotly_html("umap_plot.html"),
                style={"width": "100%", "height": "600px", "border": "none"}
            )
        ])
    ])
])

if __name__ == "__main__":
    app.run(debug=True)
    