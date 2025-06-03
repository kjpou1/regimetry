#!/bin/bash
set -e

cd "$(dirname "$0")/.." || exit 1

# 🔹 Require instrument and base config from CLI
INSTRUMENT="$1"
BASE_CONFIG="$2"

if [[ -z "$INSTRUMENT" || -z "$BASE_CONFIG" ]]; then
  echo "❌ Usage: $0 <INSTRUMENT> <BASE_CONFIG.yaml>"
  exit 1
fi

echo ""
echo "📁 Regimetry Config Exploration"
echo "🔧 Instrument:       $INSTRUMENT"
echo "📄 Base Config:      $BASE_CONFIG"
echo "📂 Script Directory: $(pwd)"
echo ""

echo "🐍 Activating Python virtual environment..."
source .venv/bin/activate
export PYTHONPATH=$(pwd)/src

echo ""
echo "🚀 Starting analyze sweep for [$INSTRUMENT]..."
echo "──────────────────────────────────────────────"

# Format: "window_size encoding_method encoding_style embedding_dim n_clusters"
# Use "None" for unset embedding_dim or encoding_style
# Format: "window_size encoding_method encoding_style embedding_dim n_clusters"
# Use "None" for unset embedding_dim or encoding_style

# CONFIGS_EX=(
#   "5 learnable None None 8"
#   "5 learnable None 64 8"
#   "5 sinusoidal interleaved 64 8"
#   "5 sinusoidal interleaved None 8"
#   "5 sinusoidal stacked 64 8"
#   "5 sinusoidal stacked None 8"
# )

# 🔧 Dynamic grid search definition
WINDOW_SIZE=5
STRIDE=1
#CLUSTER_COUNTS=(8 10 12)
CLUSTER_COUNTS=(12)
EMBEDDING_DIMS=(None 64)
ENCODING_METHODS=("learnable" "sinusoidal")
ENCODING_STYLES=("None" "interleaved" "stacked")  # Only used for sinusoidal

CONFIGS=()

for CLUSTERS in "${CLUSTER_COUNTS[@]}"; do
  for DIM in "${EMBEDDING_DIMS[@]}"; do
    for ENC_METHOD in "${ENCODING_METHODS[@]}"; do
      if [[ "$ENC_METHOD" == "learnable" ]]; then
        # No encoding style used
        CONFIGS+=("$WINDOW_SIZE $ENC_METHOD None $DIM $CLUSTERS")
      else
        for ENC_STYLE in "${ENCODING_STYLES[@]}"; do
          CONFIGS+=("$WINDOW_SIZE $ENC_METHOD $ENC_STYLE $DIM $CLUSTERS")
        done
      fi
    done
  done
done

echo ""
echo "📦 Configs to run:"
printf "  %s\n" "${CONFIGS[@]}"
echo ""

for CFG in "${CONFIGS[@]}"; do
  read -r WIN ENC_METHOD ENC_STYLE DIM CLUSTERS <<< "$CFG"

  echo ""
  echo "🔁 Running config: ws=$WIN, enc=$ENC_METHOD/$ENC_STYLE, dim=$DIM, clusters=$CLUSTERS"
  echo "──────────────────────────────────────────────"

  CMD=(
    python launch_host.py analyze
    --instrument "$INSTRUMENT"
    --base-config "$BASE_CONFIG"
    --window-size "$WIN"
    --stride 1
    --encoding-method "$ENC_METHOD"
    --n-clusters "$CLUSTERS"
    --create-dir
    --force
    --clean
  )

  if [[ "$DIM" != "None" ]]; then
    CMD+=(--embedding-dim "$DIM")
  fi

  if [[ "$ENC_METHOD" == "sinusoidal" && "$ENC_STYLE" != "None" ]]; then
    CMD+=(--encoding-style "$ENC_STYLE")
  fi

  echo "▶️  ${CMD[*]}"
  "${CMD[@]}"
  echo "✅ Finished config: ws=$WIN, dim=$DIM, clusters=$CLUSTERS"
done

echo ""
echo "🛑 Deactivating virtual environment..."
deactivate

echo ""
echo "✅ All analyze runs complete for [$INSTRUMENT]"
