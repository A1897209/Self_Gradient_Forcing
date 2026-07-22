#!/usr/bin/env bash
# Download base models, training initialization, and inference checkpoints.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
cd "$PROJECT_ROOT"

HF_REPO="${HF_REPO:-JunhaoZhuang/Self_Gradient_Forcing}"
HF_CLI="${HF_CLI:-hf}"

mkdir -p checkpoints/init/framewise checkpoints/init/chunkwise checkpoints/framewise checkpoints/chunkwise wan_models prompts

"$HF_CLI" download Wan-AI/Wan2.1-T2V-1.3B --local-dir wan_models/Wan2.1-T2V-1.3B
"$HF_CLI" download Wan-AI/Wan2.1-T2V-14B --local-dir wan_models/Wan2.1-T2V-14B

# Causal-Forcing initialization checkpoints for SGF training.
"$HF_CLI" download "$HF_REPO" init/framewise/ar_diffusion.pt --local-dir checkpoints
"$HF_CLI" download "$HF_REPO" init/framewise/causal_cd.pt --local-dir checkpoints
"$HF_CLI" download "$HF_REPO" init/framewise/causal_ode.pt --local-dir checkpoints
"$HF_CLI" download "$HF_REPO" init/chunkwise/ar_diffusion.pt --local-dir checkpoints
"$HF_CLI" download "$HF_REPO" init/chunkwise/causal_cd.pt --local-dir checkpoints
"$HF_CLI" download "$HF_REPO" init/chunkwise/causal_ode.pt --local-dir checkpoints

# Released inference checkpoints.
"$HF_CLI" download "$HF_REPO" framewise/ar/model.pt --local-dir checkpoints
"$HF_CLI" download "$HF_REPO" chunkwise/ar/model.pt --local-dir checkpoints

# Training prompt list and release metadata.
"$HF_CLI" download "$HF_REPO" vidprom_filtered_extended.txt --local-dir prompts
"$HF_CLI" download "$HF_REPO" config.json --local-dir .
"$HF_CLI" download "$HF_REPO" model_index.json --local-dir .

cat <<INFO

Done.
Framewise training initializations:
  checkpoints/init/framewise/ar_diffusion.pt
  checkpoints/init/framewise/causal_cd.pt
  checkpoints/init/framewise/causal_ode.pt
Chunkwise training initializations:
  checkpoints/init/chunkwise/ar_diffusion.pt
  checkpoints/init/chunkwise/causal_cd.pt
  checkpoints/init/chunkwise/causal_ode.pt
Framewise inference checkpoint: checkpoints/framewise/ar/model.pt
Chunkwise inference checkpoint: checkpoints/chunkwise/ar/model.pt
Training prompt list: prompts/vidprom_filtered_extended.txt
INFO
