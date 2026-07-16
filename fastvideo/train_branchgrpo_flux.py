#!/usr/bin/env python3
import argparse
import copy
import json
import os
import random
import sys
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Set

import librosa
import numpy as np
import torch
import yaml
from scipy.io import wavfile
from torch.utils.data import DataLoader
from torch.utils.checkpoint import checkpoint


@dataclass
class TreeNode:
    node_id: str
    latent: torch.Tensor
    batch_idx: int
    depth: int
    step_idx: int
    parent: Optional["TreeNode"] = None
    edge_logprob: Optional[torch.Tensor] = None
    edge_x_t: Optional[torch.Tensor] = None
    edge_x_prev: Optional[torch.Tensor] = None
    edge_t: Optional[int] = None
    reward: Optional[torch.Tensor] = None
    advantage: Optional[torch.Tensor] = None

    def __post_init__(self):
        self.children: List["TreeNode"] = []

    def add_child(self, child: "TreeNode") -> None:
        self.children.append(child)

    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def path_from_root(self) -> List["TreeNode"]:
        out = []
        cur = self
        while cur is not None:
            out.append(cur)
            cur = cur.parent
        return list(reversed(out))

    def leaf_descendants(self) -> List["TreeNode"]:
        if self.is_leaf():
            return [self]
        leaves = []
        for c in self.children:
            leaves.extend(c.leaf_descendants())
        return leaves


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_shared_codebase_root(shared_root: str) -> str:
    candidates = []
    if shared_root:
        candidates.append(os.path.expanduser(shared_root))

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    candidates.append(os.path.abspath(os.path.join(repo_root, "..", "shared_codebase")))
    candidates.append(os.path.abspath(os.path.join(repo_root, "shared_codebase")))

    for candidate in candidates:
        if os.path.isdir(candidate) and os.path.isfile(os.path.join(candidate, "datasetcode", "dataset.py")):
            return candidate

    raise FileNotFoundError(
        "Unable to locate shared_codebase. Checked: " + ", ".join(candidates)
    )


def resolve_data_path(path_value: str, shared_root: str) -> str:
    expanded = os.path.expanduser(path_value)
    if os.path.exists(expanded):
        return expanded

    marker = "shared_codebase/"
    if marker in path_value:
        suffix = path_value.split(marker, 1)[1]
        candidate = os.path.join(shared_root, suffix)
        if os.path.exists(candidate):
            return candidate

    return expanded


def add_shared_codebase_to_path(shared_root: str) -> str:
    resolved_root = resolve_shared_codebase_root(shared_root)
    if resolved_root not in sys.path:
        sys.path.insert(0, resolved_root)
    return resolved_root


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_audio(path: str, wav: np.ndarray, sr: int) -> None:
    audio_i16 = np.clip(wav * 32767.0, -32768.0, 32767.0).astype(np.int16)
    wavfile.write(path, sr, audio_i16)


def parse_split_points(branch_cfg: dict, total_steps: int) -> Set[int]:
    split_points = branch_cfg.get("tree_split_points", [])
    if isinstance(split_points, str):
        split_points = [int(x.strip()) for x in split_points.split(",") if x.strip()]
    if split_points is None:
        split_points = []
    out = set()
    for p in split_points:
        pi = int(p)
        if 0 <= pi < total_steps:
            out.add(pi)
    return out


def parse_depth_pruning(branch_cfg: dict) -> Set[int]:
    """Return exact edge depths excluded from backpropagation.

    ``depth_pruning`` accepts individual depths, while
    ``depth_pruning_ranges`` accepts inclusive ``[start, end]`` pairs.
    """
    depths = branch_cfg.get("depth_pruning", [])
    if isinstance(depths, str):
        depths = [int(x.strip()) for x in depths.split(",") if x.strip()]
    if depths is None:
        depths = []
    out = {int(x) for x in depths}

    ranges = branch_cfg.get("depth_pruning_ranges", []) or []
    for item in ranges:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ValueError(
                "each depth_pruning_ranges entry must be [start_depth, end_depth]"
            )
        start, end = (int(item[0]), int(item[1]))
        if start < 1 or end < start:
            raise ValueError(f"invalid depth pruning range: [{start}, {end}]")
        out.update(range(start, end + 1))
    return out


def wav_to_mfcc(wav: np.ndarray, cfg: dict) -> np.ndarray:
    mfcc = librosa.feature.mfcc(
        y=wav,
        sr=int(cfg["sample_rate"]),
        n_mfcc=int(cfg["mfcc_n"]),
        n_fft=int(cfg["n_fft"]),
        hop_length=int(cfg["hop_length"]),
    )
    return mfcc.astype(np.float32)


def load_bigvgan(bigvgan_dir, device, model_name="nvidia/bigvgan_22khz_80band", fmax=8000):
    """Load the BigVGAN vocoder, isolating its conflicting top-level modules.

    BigVGAN ships top-level modules named ``utils``/``env``/``activations`` that
    collide with other packages on ``sys.path``; temporarily evict them while
    importing, mirroring DanceTreeGRPO's loader.
    """
    bigvgan_dir = os.path.abspath(os.path.expanduser(bigvgan_dir))
    if not os.path.isdir(bigvgan_dir):
        raise FileNotFoundError(
            f"BigVGAN code directory not found: {bigvgan_dir}. "
            f"Set eval.bigvgan_dir to the BigVGAN repo folder."
        )

    conflicting = ["utils", "env", "activations", "meldataset", "bigvgan",
                   "alias_free_activation"]
    saved_path = list(sys.path)
    saved_modules = {name: sys.modules.pop(name, None) for name in conflicting}

    try:
        sys.path.insert(0, bigvgan_dir)
        import bigvgan as _bigvgan_module

        model = _bigvgan_module.BigVGAN.from_pretrained(model_name, use_cuda_kernel=False)
        model.h.fmax = fmax
        model.remove_weight_norm()
        model = model.eval().to(device)
    finally:
        sys.path[:] = saved_path
        for name, mod in saved_modules.items():
            if mod is not None:
                sys.modules[name] = mod
            else:
                sys.modules.pop(name, None)

    return model


@torch.no_grad()
def vocode_mel(mel: np.ndarray, vocoder, device) -> np.ndarray:
    """Vocode a mel (80, T) into a waveform using BigVGAN."""
    mel_t = torch.as_tensor(np.asarray(mel, dtype=np.float32))[None].to(device)  # (1, 80, T)
    wav = vocoder(mel_t).squeeze().detach().cpu().numpy()
    return wav.astype(np.float32)


@torch.no_grad()
def sample_with_config_steps(diffusion, motion_f, text_f, shape, branch_cfg):
    """Single-path reverse sampling using the same step count as the rollout config.

    Mirrors ``run_tree_rollout``'s stepping (``t = T - 1 - step_i`` over
    ``rollout_steps`` steps) so eval generation matches training-time sampling
    instead of running the full ``diffusion_timesteps`` schedule.
    """
    device = motion_f.device
    total_steps = min(int(branch_cfg["rollout_steps"]), int(branch_cfg["diffusion_timesteps"]))
    x = torch.randn(shape, device=device)
    B = shape[0]
    for step_i in range(total_steps):
        t = int(diffusion.T - 1 - step_i)
        if t < 0:
            break
        t_batch = torch.full((B,), t, device=device, dtype=torch.long)
        x = diffusion.p_sample(x, t_batch, motion_f, text_f)
    return x


@torch.no_grad()
def run_periodic_eval(step, cfg, diffusion, unet, cond_proj, eval_ds, vocoder, device, reward_cfg):
    """Generate a few validation samples and vocode gen + ground-truth mels to wav.

    Writes ``eval_dir/step_XXXXXXXX/sample_YYYYYYYY/{gt,gen}.wav`` so each eval
    checkpoint keeps its own audio outputs for comparison.
    """
    num_samples = min(int(cfg["eval"]["max_test_samples"]), len(eval_ds))
    if num_samples <= 0:
        return

    step_dir = os.path.join(cfg["eval"]["eval_dir"], f"step_{step:08d}")
    ensure_dir(step_dir)
    sr = int(reward_cfg["sample_rate"])

    gt_paths, gen_paths = [], []
    was_training = unet.training
    unet.eval()
    cond_proj.eval()
    try:
        for i in range(num_samples):
            sample = eval_ds[i]
            mel = sample["mel"].numpy().T  # (80, T)
            motion = sample["motion"].unsqueeze(0).to(device)
            lyrics = sample["lyrics"].unsqueeze(0).to(device)

            motion_f, text_f = cond_proj(motion, lyrics)
            out = sample_with_config_steps(
                diffusion, motion_f, text_f, (1, 80, mel.shape[1]), cfg["branchgrpo"]
            )
            gen_mel = out.squeeze(0).detach().cpu().numpy()  # (80, T)
            # de-normalize rollout output back to raw BigVGAN log-mel space.
            gen_mel = gen_mel * float(diffusion.dataset_std) + float(diffusion.dataset_mean)

            gt_wav = vocode_mel(mel, vocoder, device)
            gen_wav = vocode_mel(gen_mel, vocoder, device)

            sample_dir = os.path.join(step_dir, f"sample_{i:08d}")
            ensure_dir(sample_dir)
            gt_path = os.path.join(sample_dir, "gt.wav")
            gen_path = os.path.join(sample_dir, "gen.wav")
            save_audio(gt_path, gt_wav, sr)
            save_audio(gen_path, gen_wav, sr)
            gt_paths.append(gt_path)
            gen_paths.append(gen_path)
    finally:
        if was_training:
            unet.train()
            cond_proj.train()

    metrics = compute_eval_metrics(gt_paths, gen_paths)
    metrics["step"] = int(step)
    metrics["num_samples"] = num_samples
    metrics_path = os.path.join(step_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    printable = {k: metrics.get(k) for k in ("fad", "acoustic_similarity", "beat_f1", "js", "kl")}
    print(
        f"[eval] step={step}: wrote {num_samples} gt/gen wav pairs to {step_dir} | "
        f"metrics={printable} -> {metrics_path}"
    )


def compute_eval_metrics(gt_paths, gen_paths):
    """Score gt/gen wav pairs with the shared_codebase metrics (FAD, MFCC cosine,
    beat F1, JS/KL). Each metric is guarded so a single failure never aborts eval.
    """
    metrics = {}
    if not gt_paths or not gen_paths:
        return metrics

    try:
        from metrics.fad import compute_fad
        fad_val, _ = compute_fad(gt_paths, gen_paths)
        metrics["fad"] = float(fad_val)
    except Exception as e:
        metrics["fad"] = None
        metrics["fad_error"] = str(e)

    try:
        from metrics.acoustic_similarity import compute_pairwise_cosine
        ac = compute_pairwise_cosine(gt_paths, gen_paths)
        metrics["acoustic_similarity"] = float(np.mean(ac["per_sample"]))
    except Exception as e:
        metrics["acoustic_similarity"] = None
        metrics["acoustic_similarity_error"] = str(e)

    try:
        from metrics.beat import compute_beat_metrics
        beat = compute_beat_metrics(gt_paths, gen_paths)
        metrics["beat_f1"] = float(np.mean(beat["per_sample_f1"]))
    except Exception as e:
        metrics["beat_f1"] = None
        metrics["beat_error"] = str(e)

    try:
        from metrics.js_kl import compute_js_kl
        js_kl = compute_js_kl(gt_paths, gen_paths)
        metrics["js"] = float(js_kl["js_mean"])
        metrics["kl"] = float(js_kl["kl_mean"])
    except Exception as e:
        metrics["js"] = None
        metrics["kl"] = None
        metrics["js_kl_error"] = str(e)

    return metrics


def gaussian_logprob(sample: torch.Tensor, mean: torch.Tensor, var: torch.Tensor) -> torch.Tensor:
    var = torch.clamp(var, min=1e-8)
    log_scale = 0.5 * torch.log(2.0 * torch.as_tensor(np.pi, device=sample.device) * var)
    ll = -0.5 * ((sample - mean) ** 2) / var - log_scale
    return ll.flatten(1).mean(dim=1)


def _transition_stats(diffusion, unet, x_t, t_batch, motion_f, text_f):
    beta_t = diffusion.betas[t_batch]
    alpha_t = diffusion.alphas[t_batch]
    alpha_bar_t = diffusion.alpha_bars[t_batch]
    while beta_t.dim() < x_t.dim():
        beta_t = beta_t[..., None]
        alpha_t = alpha_t[..., None]
        alpha_bar_t = alpha_bar_t[..., None]
    eps = unet(x_t, t_batch, motion_f, text_f)
    mean = (1.0 / alpha_t.sqrt()) * (x_t - (beta_t / (1.0 - alpha_bar_t).sqrt()) * eps)
    mask = (t_batch > 0).view(-1, 1, 1).float()
    return mean, beta_t, mask


def _sample_from_mean(mean, beta_t, mask, noise):
    return mean + beta_t.sqrt() * noise * mask


def run_tree_rollout(diffusion, unet, motion_f, text_f, sample_shape, branch_cfg: dict):
    device = sample_shape.device
    bsz = sample_shape.shape[0]
    total_steps = min(int(branch_cfg["rollout_steps"]), int(branch_cfg["diffusion_timesteps"]))
    split_points = parse_split_points(branch_cfg, total_steps)

    num_generations = int(branch_cfg["num_generations"])
    branch_correlation = float(branch_cfg.get("branch_correlation", 1.0))
    if branch_correlation < 0:
        raise ValueError("branch_correlation must be non-negative")

    roots = []
    current_nodes = []
    nodes_by_depth: Dict[int, List[TreeNode]] = defaultdict(list)

    for bi in range(bsz):
        root = TreeNode(
            node_id=f"root_b{bi}",
            latent=torch.randn_like(sample_shape[bi : bi + 1]),
            batch_idx=bi,
            depth=0,
            step_idx=0,
            parent=None,
        )
        roots.append(root)
        current_nodes.append(root)
        nodes_by_depth[0].append(root)

    for step_i in range(total_steps):
        t = int(diffusion.T - 1 - step_i)
        if t < 0:
            break
        should_split = step_i in split_points

        x_t = torch.cat([n.latent for n in current_nodes], dim=0)
        t_batch = torch.full((x_t.shape[0],), t, device=device, dtype=torch.long)
        batch_ids = torch.as_tensor([n.batch_idx for n in current_nodes], device=device, dtype=torch.long)
        motion_step = motion_f[batch_ids]
        text_step = text_f[batch_ids]

        with torch.no_grad():
            mean, beta_t, mask = _transition_stats(diffusion, unet, x_t, t_batch, motion_step, text_step)

        next_nodes = []
        for ni, node in enumerate(current_nodes):
            parent_mean = mean[ni : ni + 1]
            parent_beta = beta_t[ni : ni + 1]
            parent_mask = mask[ni : ni + 1]
            parent_x_t = x_t[ni : ni + 1]

            child_count = num_generations if should_split else 1
            # Equation (2): fresh shared noise at each split plus independent
            # innovations. Normalization preserves N(0, I) child marginals.
            shared_noise = torch.randn_like(parent_mean) if should_split else None
            for ci in range(child_count):
                if should_split:
                    innovation = torch.randn_like(parent_mean)
                    noise = (shared_noise + branch_correlation * innovation) / np.sqrt(
                        1.0 + branch_correlation**2
                    )
                else:
                    noise = torch.randn_like(parent_mean)
                x_prev = _sample_from_mean(parent_mean, parent_beta, parent_mask, noise)
                lp = gaussian_logprob(x_prev.detach(), parent_mean.detach(), parent_beta.detach())

                child = TreeNode(
                    node_id=f"{node.node_id}_d{node.depth + 1}_c{ci}",
                    latent=x_prev.detach(),
                    batch_idx=node.batch_idx,
                    depth=node.depth + 1,
                    step_idx=step_i + 1,
                    parent=node,
                    edge_logprob=lp.detach()[0],
                    edge_x_t=parent_x_t.detach(),
                    edge_x_prev=x_prev.detach(),
                    edge_t=t,
                )
                node.add_child(child)
                next_nodes.append(child)

        current_nodes = next_nodes
        for n in current_nodes:
            nodes_by_depth[n.depth].append(n)

    leaf_nodes = current_nodes
    final_mel = torch.cat([n.latent for n in leaf_nodes], dim=0)
    return roots, leaf_nodes, nodes_by_depth, final_mel

def assign_tree_rewards(
    roots: List[TreeNode],
    leaf_nodes: List[TreeNode],
    leaf_rewards: torch.Tensor,
    prob_weighted: bool,
):
    for i, leaf in enumerate(leaf_nodes):
        leaf.reward = leaf_rewards[i]

    all_nodes = []
    for r in roots:
        stack = [r]
        while stack:
            n = stack.pop()
            all_nodes.append(n)
            stack.extend(n.children)

    all_nodes_sorted = sorted(all_nodes, key=lambda x: x.depth, reverse=True)

    for n in all_nodes_sorted:
        if n.is_leaf():
            continue
        child_rewards = torch.stack([c.reward for c in n.children], dim=0)
        if prob_weighted:
            logits = torch.stack([c.edge_logprob for c in n.children], dim=0).to(child_rewards.device)
            w = torch.softmax(logits, dim=0)
            n.reward = (w * child_rewards).sum()
        else:
            n.reward = child_rewards.mean()


def compute_depthwise_advantages(
    nodes_by_depth: Dict[int, List[TreeNode]], advantage_clip: Optional[float] = None
):
    for nodes in nodes_by_depth.values():
        if len(nodes) == 1:
            nodes[0].advantage = torch.tensor(0.0, device=nodes[0].latent.device)
            continue
        rewards = torch.stack([n.reward for n in nodes], dim=0)
        std = rewards.std(unbiased=False)
        if float(std.item()) < 1e-8:
            adv = torch.zeros_like(rewards)
        else:
            adv = (rewards - rewards.mean()) / (std + 1e-8)
        if advantage_clip is not None:
            adv = adv.clamp(-float(advantage_clip), float(advantage_clip))
        for i, node in enumerate(nodes):
            node.advantage = adv[i]


def select_leaves_for_backprop(
    leaf_nodes: List[TreeNode], mode: str, extreme_b: int
) -> List[TreeNode]:
    """Width-prune only after all leaf rewards have been fused and normalized."""
    if mode == "none":
        return leaf_nodes
    if mode == "parent_top1":
        groups = defaultdict(list)
        for leaf in leaf_nodes:
            split_parent = None
            for node in reversed(leaf.path_from_root()[1:]):
                if len(node.parent.children) > 1:
                    split_parent = node.parent
                    break
            key = split_parent.node_id if split_parent is not None else leaf.node_id
            groups[key].append(leaf)
        return [max(group, key=lambda n: float(n.reward.item())) for group in groups.values()]
    if mode == "extreme":
        b = max(1, int(extreme_b))
        ordered = sorted(leaf_nodes, key=lambda n: float(n.reward.item()))
        selected = ordered[:b] + ordered[-b:]
        return list({node.node_id: node for node in selected}.values())
    raise ValueError(f"unknown width_pruning_mode: {mode}")


def gather_backprop_edges(
    leaf_nodes: List[TreeNode], depth_pruning: Set[int]
) -> List[TreeNode]:
    # A shared-prefix transition is one tree edge, not one copy per leaf path.
    edges = {}
    for leaf in leaf_nodes:
        for node in leaf.path_from_root()[1:]:
            if node.depth not in depth_pruning:
                edges[node.node_id] = node
    return list(edges.values())


def recompute_edge_logprobs(
    diffusion, unet, motion_f, text_f, edges: List[TreeNode], use_checkpoint: bool = False
):
    device = motion_f.device
    out = []
    for edge_node in edges:
        x_t = edge_node.edge_x_t.to(device)
        x_prev = edge_node.edge_x_prev.to(device)
        t_batch = torch.full((1,), int(edge_node.edge_t), device=device, dtype=torch.long)
        motion = motion_f[edge_node.batch_idx : edge_node.batch_idx + 1]
        text = text_f[edge_node.batch_idx : edge_node.batch_idx + 1]

        if use_checkpoint:
            def _transition(x_in, motion_in, text_in, _t=t_batch):
                mean_, beta_, _ = _transition_stats(
                    diffusion, unet, x_in, _t, motion_in, text_in
                )
                return mean_, beta_

            mean, beta_t = checkpoint(
                _transition, x_t, motion, text, use_reentrant=False
            )
        else:
            mean, beta_t, _ = _transition_stats(
                diffusion, unet, x_t, t_batch, motion, text
            )
        out.append(gaussian_logprob(x_prev, mean, beta_t)[0])
    return torch.stack(out, dim=0)

def main() -> None:
    parser = argparse.ArgumentParser(description="Single-GPU lyrics+dance BranchGRPO training (UNet, full tree-split)")
    parser.add_argument("--config", type=str, default="config_lyrics2music_branchgrpo.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    shared_root = add_shared_codebase_to_path(cfg["paths"].get("shared_codebase_root", ""))
    cfg["paths"]["shared_codebase_root"] = shared_root
    cfg["paths"]["train_npz_dir"] = resolve_data_path(cfg["paths"]["train_npz_dir"], shared_root)
    cfg["paths"]["val_npz_dir"] = resolve_data_path(cfg["paths"]["val_npz_dir"], shared_root)
    cfg["paths"]["test_npz_dir"] = resolve_data_path(cfg["paths"]["test_npz_dir"], shared_root)

    from datasetcode.dataset import MelDataset
    from models.adan import Adan
    from models.diffusion import GaussianDiffusion
    from models.embedding import CondProjection
    from models.unet1d_ultimate import UNet1D_ultimate
    from metrics import fad as fad_module

    set_seed(int(cfg["experiment"]["seed"]))

    device = torch.device(cfg["train"]["device"] if torch.cuda.is_available() else "cpu")
    out_dir = cfg["experiment"]["output_dir"]
    ckpt_dir = os.path.join(out_dir, "checkpoints")
    tmp_audio_dir = os.path.join(out_dir, "tmp_audio")
    ensure_dir(out_dir)
    ensure_dir(ckpt_dir)
    ensure_dir(tmp_audio_dir)

    train_ds = MelDataset(cfg["paths"]["train_npz_dir"], align_mode="interp")
    loader = DataLoader(
        train_ds,
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=True,
        num_workers=int(cfg["train"]["num_workers"]),
        collate_fn=MelDataset.collate_fn,
        pin_memory=True,
    )

    cond_proj = CondProjection(
        motion_dim=78 * 3,
        text_dim=768,
        out_dim=int(cfg["model"]["cond_dim"]),
    ).to(device)

    unet = UNet1D_ultimate(
        in_dim=80,
        base_dim=int(cfg["model"]["base_dim"]),
        dim_mults=tuple(int(x) for x in cfg["model"]["dim_mults"]),
        cond_dim=int(cfg["model"]["cond_dim"]),
        time_emb_dim=int(cfg["model"]["time_emb_dim"]),
        num_res_blocks=int(cfg["model"]["num_res_blocks"]),
        mid_blocks=int(cfg["model"]["mid_blocks"]),
        attn_heads=int(cfg["model"]["attn_heads"]),
    ).to(device)

    # Load the pretrained policy checkpoint (GRPO initialization). Prefer EMA
    # weights and reuse the mel normalization stats saved in the checkpoint,
    # matching shared_codebase/sample.py and train.py.
    init_ckpt_path = cfg["train"].get("init_checkpoint", None)
    dataset_mean, dataset_std = 0.0, 1.0
    if init_ckpt_path:
        init_ckpt = torch.load(os.path.expanduser(init_ckpt_path), map_location=device)
        if "ema_unet" in init_ckpt or "ema_cond_proj" in init_ckpt:
            print(f"[init] loading EMA weights from {init_ckpt_path}")
            unet.load_state_dict(init_ckpt.get("ema_unet", init_ckpt["unet"]), strict=False)
            cond_proj.load_state_dict(init_ckpt.get("ema_cond_proj", init_ckpt["cond_proj"]), strict=False)
        else:
            print(f"[init] loading weights from {init_ckpt_path}")
            unet.load_state_dict(init_ckpt["unet"], strict=True)
            cond_proj.load_state_dict(init_ckpt["cond_proj"], strict=True)
        if init_ckpt.get("dataset_mean", None) is not None:
            dataset_mean = float(init_ckpt["dataset_mean"])
        if init_ckpt.get("dataset_std", None) is not None:
            dataset_std = float(init_ckpt["dataset_std"])
        print(f"[init] dataset mean/std = {dataset_mean} / {dataset_std}")
    else:
        print("[init] no init_checkpoint provided; training from scratch (mean=0/std=1)")

    diffusion = GaussianDiffusion(
        unet,
        timesteps=int(cfg["branchgrpo"]["diffusion_timesteps"]),
        device=str(device),
        dataset_mean=dataset_mean,
        dataset_std=dataset_std,
    )

    # EMA copies for stable evaluation/sampling and consistent checkpoint format,
    # matching shared_codebase/train.py.
    ema_decay = float(cfg["train"].get("ema_decay", 0.9998))
    ema_unet = copy.deepcopy(unet)
    ema_cond_proj = copy.deepcopy(cond_proj)
    for p in ema_unet.parameters():
        p.requires_grad = False
    for p in ema_cond_proj.parameters():
        p.requires_grad = False

    # BigVGAN neural vocoder, used for both the reward audio and periodic eval.
    bigvgan_dir = cfg.get("eval", {}).get("bigvgan_dir", os.path.join(shared_root, "BigVGAN"))
    vocoder = load_bigvgan(bigvgan_dir, device)

    optim = Adan(
        list(unet.parameters()) + list(cond_proj.parameters()),
        lr=float(cfg["train"]["lr"]),
        weight_decay=float(cfg["train"]["weight_decay"]),
    )

    step = 0
    max_steps = int(cfg["train"]["max_steps"])
    clip_range = float(cfg["train"]["clip_range"])
    ppo_epochs = int(cfg["train"]["ppo_epochs"])
    kl_beta = float(cfg["train"]["kl_beta"])
    grad_checkpointing = bool(cfg["train"].get("grad_checkpointing", False))

    reward_cfg = cfg["reward"]
    w_fad = float(reward_cfg["w_fad"])
    w_mfcc = float(reward_cfg["w_mfcc"])

    depth_pruning = parse_depth_pruning(cfg["branchgrpo"])
    tree_prob_weighted = bool(cfg["branchgrpo"].get("tree_prob_weighted", False))

    # Periodic evaluation on the validation split: generate a few samples and
    # vocode gen + gt mels to wav (reuses the vocoder loaded above).
    eval_cfg = cfg.get("eval", {}) or {}
    eval_every_steps = int(eval_cfg.get("eval_every_steps", 0))
    eval_enabled = eval_every_steps > 0 and int(eval_cfg.get("max_test_samples", 0)) > 0
    eval_ds = None
    if eval_enabled:
        try:
            eval_ds = MelDataset(cfg["paths"]["val_npz_dir"], align_mode="interp")
            ensure_dir(eval_cfg["eval_dir"])
            print(
                f"[eval] periodic eval enabled: every {eval_every_steps} steps, "
                f"{min(int(eval_cfg['max_test_samples']), len(eval_ds))} val samples, vocoder=BigVGAN"
            )
        except Exception as e:  # never let eval setup crash training
            print(f"[eval] disabled (failed to initialize: {e})")
            eval_enabled = False

    train_log = []

    for epoch in range(int(cfg["train"]["epochs"])):
        for batch in loader:
            if step >= max_steps:
                break

            gt_mel = batch["mel"].permute(0, 2, 1).to(device)
            motion = batch["motion"].to(device)
            lyrics = batch["lyrics"].to(device)

            motion_f, text_f = cond_proj(motion, lyrics)

            roots, leaf_nodes, nodes_by_depth, final_mel = run_tree_rollout(
                diffusion,
                unet,
                motion_f,
                text_f,
                sample_shape=torch.zeros_like(gt_mel),
                branch_cfg=cfg["branchgrpo"],
            )

            rewards = []
            for i, leaf in enumerate(leaf_nodes):
                gt_idx = leaf.batch_idx
                gt_mel_np = gt_mel[gt_idx].detach().cpu().numpy()
                # rollout output is in normalized space; de-normalize to raw
                # BigVGAN log-mel space before vocoding (gt mel is already raw).
                gen_mel_np = final_mel[i].detach().cpu().numpy() * dataset_std + dataset_mean

                gt_wav = vocode_mel(gt_mel_np, vocoder, device)
                gen_wav = vocode_mel(gen_mel_np, vocoder, device)

                gt_path = os.path.join(tmp_audio_dir, f"step{step}_leaf{i}_gt.wav")
                gen_path = os.path.join(tmp_audio_dir, f"step{step}_leaf{i}_gen.wav")
                save_audio(gt_path, gt_wav, int(reward_cfg["sample_rate"]))
                save_audio(gen_path, gen_wav, int(reward_cfg["sample_rate"]))

                gt_emb = fad_module._get_panns_embedding(gt_path)
                gen_emb = fad_module._get_panns_embedding(gen_path)
                # FAD-style score: negative (non-squared) L2 distance between the
                # gt/gen PANNs embeddings (matches DanceTreeGRPO per-sample FAD).
                fad_score = -float(np.linalg.norm(gen_emb - gt_emb))

                # MFCC score: cosine similarity between the mean MFCC vectors
                # (matches DanceTreeGRPO MFCC reward). Higher = more similar.
                gt_mfcc = wav_to_mfcc(gt_wav, reward_cfg).mean(axis=1)
                gen_mfcc = wav_to_mfcc(gen_wav, reward_cfg).mean(axis=1)
                denom = float(np.linalg.norm(gt_mfcc) * np.linalg.norm(gen_mfcc)) + 1e-8
                mfcc_score = float(np.dot(gt_mfcc, gen_mfcc) / denom)

                rewards.append(w_fad * fad_score + w_mfcc * mfcc_score)

            rewards_t = torch.as_tensor(rewards, device=device, dtype=torch.float32)

            assign_tree_rewards(roots, leaf_nodes, rewards_t, tree_prob_weighted)
            advantage_clip = cfg["branchgrpo"].get("advantage_clip")
            compute_depthwise_advantages(nodes_by_depth, advantage_clip)

            width_mode = str(cfg["branchgrpo"].get("width_pruning_mode", "none"))
            selected_leaves = select_leaves_for_backprop(
                leaf_nodes,
                width_mode,
                int(cfg["branchgrpo"].get("width_pruning_extreme_b", 1)),
            )
            edges = gather_backprop_edges(selected_leaves, depth_pruning)
            if not edges:
                raise RuntimeError("pruning removed every edge from the GRPO update")
            edge_adv = torch.stack([edge.advantage for edge in edges]).detach()
            edge_adv = edge_adv * float(cfg["branchgrpo"].get("advantage_scale", 1.0))
            old_edge_lp = torch.stack([edge.edge_logprob for edge in edges]).to(device)

            for update_idx in range(ppo_epochs):
                if update_idx > 0:
                    # Rebuild the conditioning graph after the previous backward.
                    motion_f, text_f = cond_proj(motion, lyrics)
                new_edge_lp = recompute_edge_logprobs(
                    diffusion,
                    unet,
                    motion_f,
                    text_f,
                    edges,
                    use_checkpoint=grad_checkpointing,
                )

                log_ratio = new_edge_lp - old_edge_lp
                ratio = torch.exp(log_ratio)
                obj1 = ratio * edge_adv
                obj2 = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range) * edge_adv
                ppo_loss = -torch.mean(torch.min(obj1, obj2))

                # Sample-based forward-KL approximation, non-negative in expectation.
                kl_term = torch.mean(torch.exp(log_ratio) - 1.0 - log_ratio)
                loss = ppo_loss + kl_beta * kl_term

                optim.zero_grad(set_to_none=True)
                loss.backward()
                parameters = list(unet.parameters()) + list(cond_proj.parameters())
                grad_clip = cfg["train"]["grad_clip"]
                if grad_clip is not None:
                    grad_norm = torch.nn.utils.clip_grad_norm_(parameters, float(grad_clip))
                else:
                    grad_norm = torch.linalg.vector_norm(
                        torch.stack([p.grad.detach().norm() for p in parameters if p.grad is not None])
                    )
                optim.step()

            # update EMA after optimizer step(s), matching shared_codebase.
            with torch.no_grad():
                for ema_p, p in zip(ema_unet.parameters(), unet.parameters()):
                    ema_p.data.mul_(ema_decay).add_(p.data * (1.0 - ema_decay))
                for ema_p, p in zip(ema_cond_proj.parameters(), cond_proj.parameters()):
                    ema_p.data.mul_(ema_decay).add_(p.data * (1.0 - ema_decay))

            if step % int(cfg["train"]["log_every_steps"]) == 0:
                # Measure policy movement after the update. With one PPO epoch,
                # pre-update ratios are identically 1 and are not diagnostic.
                with torch.no_grad():
                    diagnostic_motion_f, diagnostic_text_f = cond_proj(motion, lyrics)
                    diagnostic_lp = recompute_edge_logprobs(
                        diffusion,
                        unet,
                        diagnostic_motion_f,
                        diagnostic_text_f,
                        edges,
                    )
                    diagnostic_log_ratio = diagnostic_lp - old_edge_lp
                    diagnostic_ratio = torch.exp(diagnostic_log_ratio)
                    diagnostic_kl = torch.mean(
                        diagnostic_ratio - 1.0 - diagnostic_log_ratio
                    )
                avg_reward = float(np.mean(rewards)) if len(rewards) > 0 else 0.0
                total_nodes = sum(len(v) for v in nodes_by_depth.values())
                avg_ratio = float(diagnostic_ratio.mean().item())
                reward_std = float(np.std(rewards)) if rewards else 0.0
                adv_abs = float(edge_adv.abs().mean().item())
                ratio_std = float(diagnostic_ratio.std(unbiased=False).item())
                clip_fraction = float(
                    ((diagnostic_ratio - 1.0).abs() > clip_range).float().mean().item()
                )
                approx_kl = float(diagnostic_kl.item())
                grad_norm_value = float(grad_norm.item())
                print(
                    f"epoch={epoch} step={step} loss={float(loss.item()):.6f} "
                    f"reward={avg_reward:.6f} leaves={len(leaf_nodes)} "
                    f"nodes={total_nodes} edges={len(edges)} ratio={avg_ratio:.4f} "
                    f"ratio_std={ratio_std:.4f} kl={approx_kl:.6f} "
                    f"grad_norm={grad_norm_value:.4f} reward_std={reward_std:.4f} "
                    f"clip_frac={clip_fraction:.4f} adv_abs={adv_abs:.4f} "
                    f"split_points={sorted(parse_split_points(cfg['branchgrpo'], int(cfg['branchgrpo']['rollout_steps'])))}"
                )
                train_log.append(
                    {
                        "epoch": epoch,
                        "step": step,
                        "loss": float(loss.item()),
                        "avg_reward": avg_reward,
                        "reward_std": reward_std,
                        "num_leaves": len(leaf_nodes),
                        "selected_leaves": len(selected_leaves),
                        "total_nodes": total_nodes,
                        "backprop_edges": len(edges),
                        "avg_ratio": avg_ratio,
                        "ratio_std": ratio_std,
                        "approx_kl": approx_kl,
                        "grad_norm": grad_norm_value,
                        "clip_fraction": clip_fraction,
                        "mean_abs_advantage": adv_abs,
                        "w_fad": w_fad,
                        "w_mfcc": w_mfcc,
                    }
                )

            if step > 0 and step % int(cfg["train"]["save_every_steps"]) == 0:
                ckpt_path = os.path.join(ckpt_dir, f"ckpt_step_{step}.pt")
                torch.save(
                    {
                        "step": step,
                        "epoch": epoch,
                        "unet": unet.state_dict(),
                        "cond_proj": cond_proj.state_dict(),
                        "optim": optim.state_dict(),
                        "dataset_mean": getattr(diffusion, "dataset_mean", None),
                        "dataset_std": getattr(diffusion, "dataset_std", None),
                        "ema_unet": ema_unet.state_dict(),
                        "ema_cond_proj": ema_cond_proj.state_dict(),
                    },
                    ckpt_path,
                )
                print(f"saved checkpoint: {ckpt_path}")

            if eval_enabled and step > 0 and step % eval_every_steps == 0:
                try:
                    run_periodic_eval(
                        step,
                        cfg,
                        diffusion,
                        unet,
                        cond_proj,
                        eval_ds,
                        vocoder,
                        device,
                        reward_cfg,
                    )
                except Exception as e:  # never let eval crash training
                    print(f"[eval] step={step} failed: {e}")

            step += 1

        if step >= max_steps:
            break

    final_ckpt = os.path.join(ckpt_dir, "ckpt_final.pt")
    torch.save(
        {
            "step": step,
            "epoch": epoch,
            "unet": unet.state_dict(),
            "cond_proj": cond_proj.state_dict(),
            "optim": optim.state_dict(),
            "dataset_mean": getattr(diffusion, "dataset_mean", None),
            "dataset_std": getattr(diffusion, "dataset_std", None),
            "ema_unet": ema_unet.state_dict(),
            "ema_cond_proj": ema_cond_proj.state_dict(),
        },
        final_ckpt,
    )

    with open(os.path.join(out_dir, "train_log.json"), "w", encoding="utf-8") as f:
        json.dump(train_log, f, indent=2)

    print(f"training finished, final checkpoint: {final_ckpt}")


if __name__ == "__main__":
    main()
