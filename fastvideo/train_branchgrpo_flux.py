#!/usr/bin/env python3
import argparse
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
    depths = branch_cfg.get("depth_pruning", [])
    if isinstance(depths, str):
        depths = [int(x.strip()) for x in depths.split(",") if x.strip()]
    if depths is None:
        depths = []
    return {int(x) for x in depths}


def mel_to_wav(mel: np.ndarray, cfg: dict) -> np.ndarray:
    mel_db = mel.astype(np.float32)
    mel_power = librosa.db_to_power(np.clip(mel_db, -80.0, 20.0))
    wav = librosa.feature.inverse.mel_to_audio(
        mel_power,
        sr=int(cfg["sample_rate"]),
        n_fft=int(cfg["n_fft"]),
        hop_length=int(cfg["hop_length"]),
        win_length=int(cfg["win_length"]),
        n_iter=int(cfg["griffin_lim_iters"]),
    )
    peak = np.max(np.abs(wav))
    if peak > 1e-8:
        wav = wav / peak
    return wav.astype(np.float32)


def wav_to_mfcc(wav: np.ndarray, cfg: dict) -> np.ndarray:
    mfcc = librosa.feature.mfcc(
        y=wav,
        sr=int(cfg["sample_rate"]),
        n_mfcc=int(cfg["mfcc_n"]),
        n_fft=int(cfg["n_fft"]),
        hop_length=int(cfg["hop_length"]),
    )
    return mfcc.astype(np.float32)


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


def _sample_from_mean(mean, beta_t, mask, noise_scale=1.0):
    noise = torch.randn_like(mean)
    return mean + beta_t.sqrt() * noise * mask * noise_scale


def _width_prune_for_step(
    parents: List[TreeNode],
    candidates: List[TreeNode],
    mode: int,
    ratio: float,
) -> List[TreeNode]:
    if mode <= 0:
        return candidates

    ratio = float(max(0.0, min(1.0, ratio)))
    if len(candidates) == 0:
        return candidates

    if mode == 1:
        # Per-parent local prune.
        out = []
        parent_to_children = defaultdict(list)
        for c in candidates:
            parent_to_children[c.parent.node_id].append(c)
        for p in parents:
            ch = parent_to_children.get(p.node_id, [])
            if not ch:
                continue
            keep = max(1, int(np.ceil(len(ch) * ratio)))
            ch_sorted = sorted(ch, key=lambda n: float(n.edge_logprob.item()), reverse=True)
            out.extend(ch_sorted[:keep])
        return out

    if mode == 2:
        # Global keep: top-K only.
        keep = max(1, int(np.ceil(len(candidates) * ratio)))
        return sorted(candidates, key=lambda n: float(n.edge_logprob.item()), reverse=True)[:keep]

    return candidates


def run_tree_rollout(diffusion, unet, motion_f, text_f, sample_shape, branch_cfg: dict):
    device = sample_shape.device
    bsz = sample_shape.shape[0]
    total_steps = min(int(branch_cfg["rollout_steps"]), int(branch_cfg["diffusion_timesteps"]))
    split_points = parse_split_points(branch_cfg, total_steps)

    num_generations = int(branch_cfg["num_generations"])
    split_noise_scale = float(branch_cfg.get("tree_split_noise_scale", 1.0))
    width_mode = int(branch_cfg.get("width_pruning_mode", 0))
    width_ratio = float(branch_cfg.get("width_pruning_ratio", 1.0))

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
            created = []
            for ci in range(child_count):
                scale = split_noise_scale if should_split else 1.0
                x_prev = _sample_from_mean(parent_mean, parent_beta, parent_mask, noise_scale=scale)
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
                created.append(child)

            if should_split and width_mode > 0:
                kept = _width_prune_for_step([node], created, width_mode, width_ratio)
                next_nodes.extend(kept)
            else:
                next_nodes.extend(created)

        if should_split and width_mode == 2:
            # Optional global prune after local expansion.
            next_nodes = _width_prune_for_step(current_nodes, next_nodes, width_mode, width_ratio)

        current_nodes = next_nodes
        for n in current_nodes:
            nodes_by_depth[n.depth].append(n)

    leaf_nodes = current_nodes

    old_path_logprobs = []
    for leaf in leaf_nodes:
        path = leaf.path_from_root()[1:]
        if len(path) == 0:
            old_path_logprobs.append(torch.tensor(0.0, device=device))
            continue
        lp = torch.stack([p.edge_logprob.to(device) for p in path], dim=0).sum()
        old_path_logprobs.append(lp)

    old_path_logprobs = torch.stack(old_path_logprobs, dim=0)
    final_mel = torch.cat([n.latent for n in leaf_nodes], dim=0)
    return roots, leaf_nodes, nodes_by_depth, old_path_logprobs, final_mel


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


def compute_depthwise_advantages(nodes_by_depth: Dict[int, List[TreeNode]]):
    for depth, nodes in nodes_by_depth.items():
        if len(nodes) == 1:
            nodes[0].advantage = torch.tensor(0.0, device=nodes[0].latent.device)
            continue
        rewards = torch.stack([n.reward for n in nodes], dim=0)
        mean = rewards.mean()
        std = rewards.std(unbiased=False)
        if float(std.item()) < 1e-8:
            adv = torch.zeros_like(rewards)
        else:
            adv = (rewards - mean) / (std + 1e-8)
        for i, n in enumerate(nodes):
            n.advantage = adv[i]


def gather_leaf_advantages(leaf_nodes: List[TreeNode], depth_pruning: Set[int]) -> torch.Tensor:
    out = []
    for leaf in leaf_nodes:
        path_nodes = leaf.path_from_root()[1:]
        path_adv = [n.advantage for n in path_nodes if n.depth not in depth_pruning]
        if len(path_adv) == 0:
            out.append(torch.tensor(0.0, device=leaf.latent.device))
        else:
            out.append(torch.stack(path_adv, dim=0).mean())
    return torch.stack(out, dim=0)


def recompute_leaf_path_logprobs(diffusion, unet, motion_f, text_f, leaf_nodes: List[TreeNode], depth_pruning: Set[int]):
    device = motion_f.device
    out = []
    for leaf in leaf_nodes:
        path = leaf.path_from_root()[1:]
        lp_terms = []
        for edge_node in path:
            if edge_node.depth in depth_pruning:
                continue
            x_t = edge_node.edge_x_t.to(device)
            x_prev = edge_node.edge_x_prev.to(device)
            t_batch = torch.full((1,), int(edge_node.edge_t), device=device, dtype=torch.long)
            m = motion_f[edge_node.batch_idx : edge_node.batch_idx + 1]
            tx = text_f[edge_node.batch_idx : edge_node.batch_idx + 1]

            mean, beta_t, _ = _transition_stats(diffusion, unet, x_t, t_batch, m, tx)
            lp = gaussian_logprob(x_prev, mean, beta_t)
            lp_terms.append(lp[0])

        if len(lp_terms) == 0:
            out.append(torch.tensor(0.0, device=device))
        else:
            out.append(torch.stack(lp_terms, dim=0).sum())

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

    diffusion = GaussianDiffusion(
        unet,
        timesteps=int(cfg["branchgrpo"]["diffusion_timesteps"]),
        device=str(device),
    )

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

    reward_cfg = cfg["reward"]
    w_fad = float(reward_cfg["w_fad"])
    w_mfcc = float(reward_cfg["w_mfcc"])

    depth_pruning = parse_depth_pruning(cfg["branchgrpo"])
    tree_prob_weighted = bool(cfg["branchgrpo"].get("tree_prob_weighted", False))

    train_log = []

    for epoch in range(int(cfg["train"]["epochs"])):
        for batch in loader:
            if step >= max_steps:
                break

            gt_mel = batch["mel"].permute(0, 2, 1).to(device)
            motion = batch["motion"].to(device)
            lyrics = batch["lyrics"].to(device)

            motion_f, text_f = cond_proj(motion, lyrics)

            roots, leaf_nodes, nodes_by_depth, old_lp_sum, final_mel = run_tree_rollout(
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
                gen_mel_np = final_mel[i].detach().cpu().numpy()

                gt_wav = mel_to_wav(gt_mel_np, reward_cfg)
                gen_wav = mel_to_wav(gen_mel_np, reward_cfg)

                gt_path = os.path.join(tmp_audio_dir, f"step{step}_leaf{i}_gt.wav")
                gen_path = os.path.join(tmp_audio_dir, f"step{step}_leaf{i}_gen.wav")
                save_audio(gt_path, gt_wav, int(reward_cfg["sample_rate"]))
                save_audio(gen_path, gen_wav, int(reward_cfg["sample_rate"]))

                gt_emb = fad_module._get_panns_embedding(gt_path)
                gen_emb = fad_module._get_panns_embedding(gen_path)
                fad_score = -float(np.mean((gt_emb - gen_emb) ** 2))

                gt_mfcc = wav_to_mfcc(gt_wav, reward_cfg)
                gen_mfcc = wav_to_mfcc(gen_wav, reward_cfg)
                min_t = min(gt_mfcc.shape[1], gen_mfcc.shape[1])
                mfcc_mse = float(np.mean((gt_mfcc[:, :min_t] - gen_mfcc[:, :min_t]) ** 2))
                mfcc_score = -mfcc_mse

                rewards.append(w_fad * fad_score + w_mfcc * mfcc_score)

            rewards_t = torch.as_tensor(rewards, device=device, dtype=torch.float32)

            assign_tree_rewards(roots, leaf_nodes, rewards_t, tree_prob_weighted)
            compute_depthwise_advantages(nodes_by_depth)
            leaf_adv = gather_leaf_advantages(leaf_nodes, depth_pruning)

            for _ in range(ppo_epochs):
                new_lp_sum = recompute_leaf_path_logprobs(
                    diffusion,
                    unet,
                    motion_f,
                    text_f,
                    leaf_nodes,
                    depth_pruning,
                )

                ratio = torch.exp(new_lp_sum - old_lp_sum)
                obj1 = ratio * leaf_adv
                obj2 = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range) * leaf_adv
                ppo_loss = -torch.mean(torch.min(obj1, obj2))

                kl_term = torch.mean(old_lp_sum - new_lp_sum)
                loss = ppo_loss + kl_beta * kl_term

                optim.zero_grad(set_to_none=True)
                loss.backward()
                grad_clip = cfg["train"]["grad_clip"]
                if grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(
                        list(unet.parameters()) + list(cond_proj.parameters()),
                        float(grad_clip),
                    )
                optim.step()

            if step % int(cfg["train"]["log_every_steps"]) == 0:
                avg_reward = float(np.mean(rewards)) if len(rewards) > 0 else 0.0
                print(
                    f"epoch={epoch} step={step} loss={float(loss.item()):.6f} "
                    f"reward={avg_reward:.6f} leaves={len(leaf_nodes)} "
                    f"split_points={sorted(parse_split_points(cfg['branchgrpo'], int(cfg['branchgrpo']['rollout_steps'])))}"
                )
                train_log.append(
                    {
                        "epoch": epoch,
                        "step": step,
                        "loss": float(loss.item()),
                        "avg_reward": avg_reward,
                        "num_leaves": len(leaf_nodes),
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
                        "config": cfg,
                    },
                    ckpt_path,
                )
                print(f"saved checkpoint: {ckpt_path}")

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
            "config": cfg,
        },
        final_ckpt,
    )

    with open(os.path.join(out_dir, "train_log.json"), "w", encoding="utf-8") as f:
        json.dump(train_log, f, indent=2)

    print(f"training finished, final checkpoint: {final_ckpt}")


if __name__ == "__main__":
    main()
