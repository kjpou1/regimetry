#!/bin/bash
cd "$(dirname "$0")/.." || exit 1

# 🔹 Require instrument and base config from CLI
INSTRUMENT="$1"
BASE_CONFIG="$2"

if [[ -z "$INSTRUMENT" || -z "$BASE_CONFIG" ]]; then
  echo "❌ Usage: $0 <INSTRUMENT> <BASE_CONFIG.yaml>"
  exit 1
fi

echo "🐍 Activating Python virtual environment..."
source .venv/bin/activate
export PYTHONPATH=$(pwd)/src

echo "🚀 Starting Regimetry analyze batch for [$INSTRUMENT] with base config [$BASE_CONFIG]..."

# Format: "window_size encoding_method encoding_style embedding_dim n_clusters"
# Use "None" for unset embedding_dim or encoding_style
CONFIGS=(
#   "5 learnable None None 10"
#   "5 learnable None None 12"
#   "5 learnable None 64 10"
#   "5 learnable None 64 12"
#   "5 sinusoidal interleaved 64 10"
#   "5 sinusoidal interleaved 64 12"
#   "5 sinusoidal interleaved None 10"
#   "5 sinusoidal interleaved None 12"
#   "5 sinusoidal stacked 64 10"
#   "5 sinusoidal stacked 64 12"
#   "5 sinusoidal stacked None 10"
#   "5 sinusoidal stacked None 12"
    "5 learnable None 64 8"

)

for CFG in "${CONFIGS[@]}"; do
  read -r WIN ENC_METHOD ENC_STYLE DIM CLUSTERS <<< "$CFG"
  echo "📊 [$INSTRUMENT] ws=$WIN enc=$ENC_METHOD/$ENC_STYLE dim=$DIM clusters=$CLUSTERS"

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

  # Only add --embedding-dim if not None
  if [[ "$DIM" != "None" ]]; then
    CMD+=(--embedding-dim "$DIM")
  fi

  # Only add --encoding-style if encoding-method is sinusoidal
  if [[ "$ENC_METHOD" == "sinusoidal" && "$ENC_STYLE" != "None" ]]; then
    CMD+=(--encoding-style "$ENC_STYLE")
  fi

  echo "▶️ Running: ${CMD[*]}"
  "${CMD[@]}"
done

echo "🛑 Deactivating virtual environment..."
deactivate

echo "✅ All analyze runs complete for [$INSTRUMENT]"
