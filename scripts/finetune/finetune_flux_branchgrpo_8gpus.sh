#!/usr/bin/env bash
set -euo pipefail

# Single-GPU launcher for lyrics+dance-to-music BranchGRPO.
# Kept at the existing path for compatibility with prior automation.

CONFIG_PATH="${1:-config_lyrics2music_branchgrpo.yaml}"

python fastvideo/train_branchgrpo_flux.py --config "${CONFIG_PATH}"
