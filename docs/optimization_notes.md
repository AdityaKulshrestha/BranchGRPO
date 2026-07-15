# BranchGRPO — Possible Optimizations & Known Issues

Notes on current limitations of `fastvideo/train_branchgrpo_flux.py` and how to address
them. The "Implemented" section below is live in the code; the numbered sections after it
are recorded as future work.

## Implemented: memory & precision optimizations

Goal: cut training memory enough to **disable gradient checkpointing** (which was slowing
training with its backward recompute), **without** changing `rollout_steps` (1000) or the
leaf count (`num_generations` / split points).

### A. Shared-trunk log-prob pruning
- **Config:** `branchgrpo.prune_shared_trunk: true`
- **What:** The denoising steps *before the first tree split* form a single shared chain
  (one node per depth). Those depths are excluded from the PPO log-prob computation in
  both `run_tree_rollout` (old log-probs) and `recompute_leaf_path_logprobs` (new
  log-probs), so the PPO ratio still covers a consistent edge set. The pruned depth set is
  `1 .. min(tree_split_points)`.
- **Why it's safe:** Trunk depths have **zero advantage** (single node per depth), so they
  contribute no policy-gradient signal; the shared trunk is identical across all leaves.
- **Impact (biggest win):** Removes the trunk forwards from the autograd graph. With a
  split at 995, grad-enabled UNet forwards drop from `leaves × 1000` to
  `leaves × (1000 − 995)` — e.g. `4 × 1000 = 4000` → `4 × 5 = 20`. This is what makes
  running without gradient checkpointing feasible.
- **Explicitly not changed:** The **advantage aggregation** (`gather_leaf_advantages`)
  still averages over the *full* path, so this does not alter the advantage magnitude
  discussed elsewhere — it only touches the log-prob/ratio graph.

### B. bf16 mixed precision
- **Config:** `train.bf16: true`
- **What:** UNet forwards (rollout, PPO recompute, and periodic-eval sampling) run under
  `torch.autocast("cuda", dtype=torch.bfloat16)`. `eps` is upcast to fp32 immediately, so
  the transition mean and Gaussian log-prob math stay in fp32. Weights, optimizer state,
  and EMA remain fp32 (standard mixed precision — no `GradScaler` needed for bf16).
- **Guarded:** Falls back to fp32 with a printed warning if
  `torch.cuda.is_bf16_supported()` is false (needs Ampere/Hopper, e.g. A100/H100).
- **Impact:** ~2× lower activation/input memory for the retained forwards and faster
  matmuls. Combined with (A), the grad graph holds only the post-split forwards in bf16.

### C. Gradient checkpointing disabled
- **Config:** `train.grad_checkpointing: false`
- **What:** The checkpoint code path is retained (flag-driven) but turned off by default,
  since (A) removes the huge trunk recompute and (B) halves activation memory.
- **Impact:** No backward-time recompute → faster steps, at the cost of holding the (now
  small, ~20) post-split forwards' activations in memory.

### Combined effect
Peak activation term goes from `≈ leaves × 1000 × A_unet(fp32)` (infeasible without
checkpointing) to `≈ leaves × postSplitSteps × A_unet(bf16)` — a reduction of ~2–3 orders
of magnitude for the current config — while keeping rollout length and leaf count intact.

### Optional further win (not implemented)
- **Edge deduplication:** even post-trunk-pruning, sibling branches recompute per leaf
  path. Computing each *unique* post-split node once (instead of per leaf) would shave the
  remaining forwards further (e.g. `20 → ~unique post-split nodes`). Skipped for now
  because the trunk pruning already brings the count into a comfortable range.

## 1. Batching issue (`batch_size > 1` not supported)

### Problem
`batch_size` is fixed at `1`. Setting it higher breaks in two places:

1. **Collate cannot stack variable-length mels.**
   `MelDataset.collate_fn` (shared_codebase/datasetcode/dataset.py) uses plain
   `torch.stack` and explicitly assumes every sample already has the same time length `T`:
   ```python
   mel = torch.stack([b['mel'] for b in batch], dim=0)   # requires identical T
   ```
   Clips have different lengths (e.g. 516 vs other), so two different-length clips in one
   batch raise a size-mismatch error.

2. **Advantages mix different conditions.**
   `compute_depthwise_advantages` z-scores across **all** nodes at a depth:
   ```python
   rewards = torch.stack([n.reward for n in nodes], dim=0)   # nodes from ALL batch items
   adv = (rewards - mean) / (std + 1e-8)
   ```
   With `batch_size > 1`, `nodes_by_depth[depth]` pools nodes from different
   `(motion, lyrics)` conditions. GRPO must normalize **within** each condition's own
   generations, so cross-condition pooling contaminates the advantage estimates.

### Fix (remove the padding — use fixed-length inputs)
Rather than padding to the batch max `T` and carrying a length mask (which adds padded
frames the reward must then ignore), make every sample a **fixed-length crop** so a plain
`torch.stack` works with no padding at all:

- Add a config field, e.g. `data.sequence_length` (fixed number of mel frames).
- In the dataset / collate, crop (or resample) mel + motion + lyrics to that fixed `T`
  for every sample. All samples then share `T`, so `torch.stack` batches them directly
  and the reward sees no padded frames.
- Bucket by length only if fixed-length cropping is undesirable.

Then also fix the advantage grouping so `batch_size > 1` is correct:

- In `compute_depthwise_advantages`, group nodes by `(depth, batch_idx)` (i.e. per root)
  before computing mean/std, so each condition's generations are normalized among
  themselves.

Cost note: increasing `batch_size` multiplies the number of leaves (each leaf runs a
BigVGAN vocode + PANNs embedding), on top of the `num_generations^(num_splits)` growth
already incurred by the tree.

## 2. `kl_beta = 0` — no KL constraint currently

### Current behavior
The loss is the pure PPO surrogate:
```python
kl_term = torch.mean(old_lp_sum - new_lp_sum)
loss = ppo_loss + kl_beta * kl_term
```
With `kl_beta = 0.0`, the KL term is dropped entirely — there is **no explicit KL
penalty**. Two caveats:

1. PPO clipping (`clip_range: 0.2`) still bounds the per-update policy change, so this is
   not unconstrained. Relying solely on the clip is a valid GRPO/PPO choice.
2. This `kl_term` is `old_lp - new_lp` (rollout policy vs. current policy) — a PPO-style
   penalty, **not** a KL against a frozen reference model. No frozen reference policy is
   kept, and with `ppo_epochs: 1` the term is ~0 anyway (`new_lp == old_lp` on the single
   update).

### Fix (anchor to the pretrained init)
For standard GRPO fine-tuning that prevents drift away from the pretrained policy:

- Keep a **frozen reference model** loaded from `train.init_checkpoint` (EMA weights),
  with `requires_grad = False`.
- Compute a reference KL per denoising step (reference log-prob vs. current log-prob of
  the same transition) and add it as `kl_beta * kl_ref` to the loss.
- Set `kl_beta > 0` (e.g. `0.01`–`0.1`) to tune how tightly the policy is anchored.

This is the mechanism that actually stops the reward from steadily degrading away from
the pretrained checkpoint.
