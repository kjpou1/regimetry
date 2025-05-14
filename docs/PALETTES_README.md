
# 🎨 Palette Options for Clustering Reports

This document provides guidance on selecting color palettes for regime clustering visualizations in `regimetry`.

## 🔍 Why Palettes Matter

When visualizing cluster-based regimes (e.g., in t-SNE, UMAP, and overlay plots), **clear and distinguishable coloring** is essential for interpretability. We use seaborn palettes to ensure consistent, aesthetically pleasing visuals across both Matplotlib and Plotly outputs.

---

## ✅ Recommended Palettes

These palettes have been tested for **clarity, contrast, and consistency** across backends:

| Palette Name | Style       | Max Distinct Colors | Description                                                |
| ------------ | ----------- | ------------------- | ---------------------------------------------------------- |
| `tab10`      | **Default** | 10                  | Standard Matplotlib palette, well-balanced for general use |
| `Set1`       | Bright      | 9                   | High-contrast, bold colors—great for <9 clusters           |
| `Set2`       | Pastel      | 8                   | Soft, muted colors—good for visual comfort                 |
| `Dark2`      | Earth Tones | 8                   | Distinct and darker shades, ideal for overlays             |
| `Paired`     | Hue Pairs   | 12                  | Designed for grouped clusters (e.g., bullish/bearish)      |
| `colorblind` | Accessible  | 8                   | Colorblind-friendly, but still visually distinct           |

> ℹ️ If you specify more clusters than the palette supports, colors may cycle or interpolate — possibly reducing distinguishability.

---

## 🔧 Configuration

You can set the palette using the `report_palette` option in your YAML config:

```yaml
report_format: ["matplotlib", "plotly"]
report_palette: "Set2"
```

This controls:

* Matplotlib color maps (`ListedColormap`)
* Plotly color mappings (`color_discrete_map`)
* Overlay charts and legends

---

## 🧪 Advanced Options

Want to visualize your palette before committing?

```python
import seaborn as sns
import matplotlib.pyplot as plt

sns.palplot(sns.color_palette("Set2", 8))  # preview first 8 colors
plt.title("Set2 Palette Preview")
plt.show()
```

---

## 🚫 Avoid These for Clusters

These palettes may be beautiful but are not ideal for discrete cluster visualization:

* `rocket`, `mako`, `viridis`, `coolwarm`, etc. → **Continuous palettes**
* `pastel1`, `Pastel2` → Often too low-contrast
* `deep` or `muted` → May be indistinct in scatter overlays

---

## 📊 Best Practices

* Stick to **8–10 clusters** if you want strong visual separation.
* Use `Paired` or continuous palettes only if your use case supports it.
* Always **match `n_clusters`** to your visualization config for consistent color alignment.

