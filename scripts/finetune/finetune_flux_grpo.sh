#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:-config_lyrics2music_branchgrpo.yaml}"
CKPT_PATH="${2:-/home/anamf/aditya/dance2music/BranchGRPO/outputs/lyrics_dance2music_branchgrpo/checkpoints/ckpt_final.pt}"

python fastvideo/eval_lyrics2music.py --config "${CONFIG_PATH}" --checkpoint "${CKPT_PATH}"
