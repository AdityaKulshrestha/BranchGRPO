# BranchGRPO for Lyrics + Dance to Music

This repository is configured for a single experiment family: lyrics+dance-to-music training with BranchGRPO tree rollout mechanics, running on one GPU and reusing UNet/audio components from shared_codebase.

## What is active in this codebase

- Model stack: CondProjection + UNet1D_ultimate + GaussianDiffusion from shared_codebase modules.
- Training entry: fastvideo/train_branchgrpo_flux.py.
- Evaluation entry: fastvideo/eval_lyrics2music.py.
- One unified configuration file: config_lyrics2music_branchgrpo.yaml.
- Shared dataset roots are used directly (no separate preprocessing tree in BranchGRPO).

## Setup

1. Create or activate your environment.
2. Install dependencies:

```bash
cd /home/anamf/aditya/dance2music/BranchGRPO
pip install -r requirements.txt
```

3. Ensure the shared dataset paths exist and are readable:

- /home/anamf/aditya/dance2music/shared_codebase/npz_split/train
- /home/anamf/aditya/dance2music/shared_codebase/npz_split/val
- /home/anamf/aditya/dance2music/shared_codebase/npz_split/test

4. Ensure the local FAD checkpoint expected by shared metrics exists:

- fastvideo/../shared_codebase/metrics/audioset_tagging_cnn/checkpoints/Cnn14_mAP=0.431.pth

## Training

Run training directly:

```bash
cd /home/anamf/aditya/dance2music/BranchGRPO
python fastvideo/train_branchgrpo_flux.py --config config_lyrics2music_branchgrpo.yaml
```

Or via launcher:

```bash
bash scripts/finetune/finetune_flux_branchgrpo_8gpus.sh config_lyrics2music_branchgrpo.yaml
```

Checkpoints are saved under:

- outputs/lyrics_dance2music_branchgrpo/checkpoints

## Inference + Evaluation

Run single-process generation and metrics:

```bash
cd /home/anamf/aditya/dance2music/BranchGRPO
python fastvideo/eval_lyrics2music.py \
  --config config_lyrics2music_branchgrpo.yaml \
  --checkpoint /home/anamf/aditya/dance2music/BranchGRPO/outputs/lyrics_dance2music_branchgrpo/checkpoints/ckpt_final.pt
```

Or via launcher:

```bash
bash scripts/finetune/finetune_flux_grpo.sh \
  config_lyrics2music_branchgrpo.yaml \
  /home/anamf/aditya/dance2music/BranchGRPO/outputs/lyrics_dance2music_branchgrpo/checkpoints/ckpt_final.pt
```

Outputs:

- Sample audio pairs: outputs/lyrics_dance2music_branchgrpo/evaluation/sample_xxxxxxxx/{gt.wav,gen.wav}
- JSON metrics: outputs/lyrics_dance2music_branchgrpo/results/evaluation_results.json

## Tree-BranchGRPO mechanics

The trainer includes full tree rollout and optimization flow:

1. Split-point branching:
- At diffusion steps listed in branchgrpo.tree_split_points, each live node expands to branchgrpo.num_generations children.

2. Width pruning:
- width_pruning_mode=0: disabled.
- width_pruning_mode=1: keep top children per parent by edge log-prob.
- width_pruning_mode=2: keep global top children by edge log-prob.
- width_pruning_ratio controls retained fraction.

3. Reward assignment and upstream propagation:
- Leaf rewards are computed from generated audio.
- Internal node rewards are upstreamed from children by:
  - mean(child rewards), or
  - softmax(edge log-prob)-weighted sum if tree_prob_weighted=true.

4. Depth-wise advantages:
- Nodes at the same depth are normalized to produce node advantages.
- Leaf optimization advantage is the mean of path node advantages, optionally excluding depths listed in depth_pruning.

5. PPO update on path likelihoods:
- Old path log-prob is recorded during rollout.
- New path log-prob is recomputed with current model.
- PPO clipped ratio objective is applied per leaf path.

## Sampling mode: DDPM or DDIM

- Active sampling in both training rollout and evaluation is DDPM-style ancestral sampling.
- In training rollout, each reverse step samples x_{t-1} from mean + sqrt(beta_t)*noise.
- In evaluation, GaussianDiffusion.sample calls p_sample iteratively over timesteps.
- A DDIM helper exists in shared_codebase/models/diffusion.py (ddim_sample), but it is not used by the active train/eval scripts.

## Batching support

- Training supports batching.
- DataLoader uses train.batch_size.
- Tree rollout builds one root per batch sample and tracks batch_idx through branching.
- Reward computation maps each leaf back to its source batch item via leaf.batch_idx.

- Evaluation script currently runs one sample at a time (it loops over dataset and uses shape (1, 80, T)).
- The diffusion core supports batched sampling, but eval script is intentionally single-process/single-item for stable metric generation.

## Reward definition

Per leaf:

- FAD proxy score:
  - Convert mel to waveform.
  - Extract PANNs embedding for gt/gen wav.
  - score_fad = -mean((emb_gt - emb_gen)^2).

- MFCC score:
  - Compute MFCC(gt) and MFCC(gen).
  - score_mfcc = -MSE(MFCC_gt, MFCC_gen).

Final leaf reward:

- reward = reward.w_fad * score_fad + reward.w_mfcc * score_mfcc

These leaf rewards are then upstreamed through the rollout tree to compute node rewards and advantages.

## Main config knobs

Key fields in config_lyrics2music_branchgrpo.yaml:

- train.batch_size
- train.ppo_epochs
- train.clip_range
- branchgrpo.tree_split_points
- branchgrpo.num_generations
- branchgrpo.width_pruning_mode
- branchgrpo.width_pruning_ratio
- branchgrpo.depth_pruning
- branchgrpo.tree_prob_weighted
- reward.w_fad
- reward.w_mfcc
