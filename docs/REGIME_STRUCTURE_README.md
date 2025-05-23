# 📘 Regime Structure Analysis

This module performs **structural analysis** of clustered market regimes to help interpret their stability, volatility, and behavioral tendencies.

It is a post-clustering pipeline that builds a **regime-level summary** using Markov transitions, entropy, and stickiness — and outputs both human-readable reports and runtime strategy metadata.

---

## ✅ What This Provides

This pipeline generates:

- ✅ **Transition matrix** (Markov regime-to-regime flow)
- ✅ **Stickiness** (self-persistence of each regime)
- ✅ **Entropy** (regime stability vs. chaos)
- ✅ **Volatility flags** (based on entropy)
- ✅ **Most-likely regime transitions**
- ✅ **Strategy decision table**
- ✅ **Heatmap visualization**
- ✅ **JSON metadata for strategy filtering**

---

## 🧠 Why This Matters

Understanding regime structure allows you to:

- Filter trades in **unstable** or **noisy** regimes
- Size trades more confidently in **stable**, trending regimes
- Anticipate when regimes are likely to transition
- Provide context-aware logic for entry/exit decisions

---

## 🚀 CLI Usage

You run the structure pipeline using the `interpret` subcommand:

```bash
python launch_host.py interpret \
  --input-path artifacts/reports/EUR_USD/cluster_assignments.csv \
  --output-dir artifacts/reports/EUR_USD/ \
  --save-csv \
  --save-heatmap \
  --save-json
````

---

## 📦 Output Files

| File                            | Description                                                            |
| ------------------------------- | ---------------------------------------------------------------------- |
| `transition_matrix.csv`         | Markov probabilities of moving from one regime to another              |
| `regime_decision_table.csv`     | Full table with stickiness, entropy, volatility, and strategy guidance |
| `transition_matrix_heatmap.png` | Visual summary of regime transitions                                   |
| `regime_metadata.json`          | Minified dictionary for runtime strategy logic                         |

---

## 🧠 Metadata Explained (`regime_metadata.json`)

Each cluster ID is mapped to:

```json
{
  "2": {
    "stickiness": 0.859,
    "entropy": 0.485,
    "volatile": true,
    "note": "🟡 Moderate regime, tighten stop"
  }
}
```

### Field Definitions:

| Field        | Meaning                                              |
| ------------ | ---------------------------------------------------- |
| `stickiness` | Probability the regime stays in itself next timestep |
| `entropy`    | Entropy of transition row — higher = more disorder   |
| `volatile`   | True if entropy > 0.4 (can be tuned)                 |
| `note`       | Strategy guidance string                             |

---

## 🔁 Strategy Integration

Example usage in live decision logic:

```python
import json

with open("regime_metadata.json") as f:
    regime_metadata = json.load(f)

cluster = "4"
if regime_metadata[cluster]["volatile"]:
    suppress_entry = True
```

---

## ⚙️ How Values Are Calculated

* **Transition matrix**: Based on sequential `Cluster_ID` values
* **Stickiness**: Diagonal of the transition matrix (P(cluster → same))
* **Entropy**: Shannon entropy of transition probabilities
* **Volatile**: `True` if entropy > 0.4
* **Strategy Note**:

  * > 0.90 stickiness → `"✅ Trend-friendly, hold longer"`
  * > 0.75 → `"🟡 Moderate regime, tighten stop"`
  * otherwise → `"⚠️ Transitional, exit fast"`

---

## 📂 Recommended Directory Layout

```bash
artifacts/reports/EUR_USD/
├── cluster_assignments.csv
├── regime_decision_table.csv
├── transition_matrix.csv
├── transition_matrix_heatmap.png
├── regime_metadata.json
```
