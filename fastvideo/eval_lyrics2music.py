#!/usr/bin/env python3
import argparse
import json
import os
import sys
from glob import glob

import librosa
import numpy as np
import torch
import yaml
from scipy.io import wavfile
from tqdm import tqdm


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def add_shared_codebase_to_path(shared_root: str) -> None:
    if shared_root not in sys.path:
        sys.path.insert(0, shared_root)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_audio(path: str, wav: np.ndarray, sr: int) -> None:
    audio_i16 = np.clip(wav * 32767.0, -32768.0, 32767.0).astype(np.int16)
    wavfile.write(path, sr, audio_i16)


def mel_to_wav(mel: np.ndarray, reward_cfg: dict) -> np.ndarray:
    mel_db = mel.astype(np.float32)
    mel_power = librosa.db_to_power(np.clip(mel_db, -80.0, 20.0))
    wav = librosa.feature.inverse.mel_to_audio(
        mel_power,
        sr=int(reward_cfg["sample_rate"]),
        n_fft=int(reward_cfg["n_fft"]),
        hop_length=int(reward_cfg["hop_length"]),
        win_length=int(reward_cfg["win_length"]),
        n_iter=int(reward_cfg["griffin_lim_iters"]),
    )
    peak = np.max(np.abs(wav))
    if peak > 1e-8:
        wav = wav / peak
    return wav.astype(np.float32)


def scan_eval_dir(eval_root: str):
    samples = []
    for d in sorted(glob(os.path.join(eval_root, "sample_*"))):
        gt = os.path.join(d, "gt.wav")
        gen = os.path.join(d, "gen.wav")
        if os.path.exists(gt) and os.path.exists(gen):
            samples.append((os.path.basename(d), gt, gen))
    return samples


def main():
    parser = argparse.ArgumentParser(description="Single-process sample + evaluation for lyrics+dance BranchGRPO")
    parser.add_argument("--config", type=str, default="config_lyrics2music_branchgrpo.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    add_shared_codebase_to_path(cfg["paths"]["shared_codebase_root"])

    from datasetcode.dataset import MelDataset
    from models.diffusion import GaussianDiffusion
    from models.embedding import CondProjection
    from models.unet1d_ultimate import UNet1D_ultimate

    # Shared evaluation settings and metrics from shared_codebase.
    from metrics.acoustic_similarity import compute_pairwise_cosine as compute_acoustic_similarity
    from metrics.beat import compute_beat_metrics
    from metrics.fad import compute_fad
    from metrics.js_kl import compute_js_kl
    from metrics.ndb import compute_ndb

    device = torch.device(cfg["train"]["device"] if torch.cuda.is_available() else "cpu")

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

    ckpt = torch.load(args.checkpoint, map_location=device)
    unet.load_state_dict(ckpt["unet"], strict=True)
    cond_proj.load_state_dict(ckpt["cond_proj"], strict=True)
    unet.eval()
    cond_proj.eval()

    diffusion = GaussianDiffusion(
        unet,
        timesteps=int(cfg["branchgrpo"]["diffusion_timesteps"]),
        device=str(device),
    )

    eval_dir = cfg["eval"]["eval_dir"]
    results_dir = cfg["eval"]["results_dir"]
    ensure_dir(eval_dir)
    ensure_dir(results_dir)

    test_ds = MelDataset(cfg["paths"]["test_npz_dir"], align_mode="interp")
    max_samples = min(int(cfg["eval"]["max_test_samples"]), len(test_ds))

    for i in tqdm(range(max_samples), desc="generating"):
        sample = test_ds[i]
        mel = sample["mel"].numpy().T
        motion = sample["motion"].unsqueeze(0).to(device)
        lyrics = sample["lyrics"].unsqueeze(0).to(device)

        with torch.no_grad():
            motion_f, text_f = cond_proj(motion, lyrics)
            out = diffusion.sample((1, 80, mel.shape[1]), motion_f, text_f)
            gen_mel = out.squeeze(0).detach().cpu().numpy()

        gt_wav = mel_to_wav(mel, cfg["reward"])
        gen_wav = mel_to_wav(gen_mel, cfg["reward"])

        sample_dir = os.path.join(eval_dir, f"sample_{i:08d}")
        ensure_dir(sample_dir)
        save_audio(os.path.join(sample_dir, "gt.wav"), gt_wav, int(cfg["reward"]["sample_rate"]))
        save_audio(os.path.join(sample_dir, "gen.wav"), gen_wav, int(cfg["reward"]["sample_rate"]))

    pairs = scan_eval_dir(eval_dir)
    gt_list = [x[1] for x in pairs]
    gen_list = [x[2] for x in pairs]

    results = {
        "metadata": {
            "total_samples": len(pairs),
            "eval_dir": eval_dir,
            "checkpoint": args.checkpoint,
        },
        "batch_metrics": {},
        "per_sample_metrics": {},
    }

    # Shared-style batch metrics.
    try:
        fad_val, _ = compute_fad(gt_list, gen_list)
        results["batch_metrics"]["fad_overall"] = float(fad_val)
    except Exception as e:
        results["batch_metrics"]["fad_overall"] = None
        results["batch_metrics"]["fad_error"] = str(e)

    try:
        js_kl = compute_js_kl(gt_list, gen_list)
        results["batch_metrics"]["js_kl_overall"] = {
            "js_mean": float(js_kl["js_mean"]),
            "kl_mean": float(js_kl["kl_mean"]),
        }
    except Exception as e:
        results["batch_metrics"]["js_kl_overall"] = None
        results["batch_metrics"]["js_kl_error"] = str(e)

    try:
        ndb = compute_ndb(gt_list, gen_list, K=50)
        results["batch_metrics"]["ndb_overall"] = int(ndb["ndb"])
        results["batch_metrics"]["ndb_K"] = 50
    except Exception as e:
        results["batch_metrics"]["ndb_overall"] = None
        results["batch_metrics"]["ndb_error"] = str(e)

    # Per-sample acoustic (MFCC cosine) and beat metrics.
    acoustic_vals = []
    beat_vals = []
    for sid, gt, gen in pairs:
        per_item = {"gt": gt, "gen": gen}
        try:
            ac = compute_acoustic_similarity([gt], [gen])
            per_item["acoustic_similarity"] = float(ac["per_sample"][0])
            acoustic_vals.append(per_item["acoustic_similarity"])
        except Exception as e:
            per_item["acoustic_similarity"] = None
            per_item["acoustic_error"] = str(e)

        try:
            beat = compute_beat_metrics([gt], [gen])
            per_item["beat_f1"] = float(beat["per_sample_f1"][0])
            per_item["beat_precision"] = float(beat["per_sample_precision"][0])
            per_item["beat_recall"] = float(beat["per_sample_recall"][0])
            beat_vals.append(per_item["beat_f1"])
        except Exception as e:
            per_item["beat_f1"] = None
            per_item["beat_error"] = str(e)

        results["per_sample_metrics"][sid] = per_item

    if acoustic_vals:
        results["metadata"]["acoustic_similarity_mean"] = float(np.mean(acoustic_vals))
    if beat_vals:
        results["metadata"]["beat_f1_mean"] = float(np.mean(beat_vals))

    out_json = os.path.join(results_dir, "evaluation_results.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"evaluation done: {out_json}")


if __name__ == "__main__":
    main()
