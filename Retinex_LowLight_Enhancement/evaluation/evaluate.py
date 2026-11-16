"""
Comprehensive Evaluation and Comparison Benchmark Script.
Evaluates:
- Experiment A: Corrected Retinex Baseline
- Experiment B: Baseline + Proposed IGNRR
- Experiment C: Baseline + Refinement w/o Noise-Aware Component
- Experiment D: Baseline + Refinement w/o Illumination Guidance

Calculates all standard metrics:
- PSNR (dB)
- SSIM
- Delta E 2000 (CIEDE2000 Color Difference)
- RGB MAE
- CIELAB channel errors (L*, a*, b*)
- Average Inference Latency (ms/image)
"""

import os
import sys
import time
import argparse
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from configs.config import (
    DATASET_DIR,
    CORRECTED_BASELINE_DIR,
    IMPROVED_CHECKPOINT_DIR,
    ABLATION_CHECKPOINT_DIR,
    EXPERIMENTS_DIR,
    RESULTS_DIR
)
from utils.dataset import LOLDataset
from utils.checkpoint import load_checkpoint
from utils.metrics import compute_psnr, compute_ssim, compute_color_metrics
from models.decom_net import DecomNet
from models.relight_net import RelightNet
from models.ignrr_net import (
    build_ignrr_full,
    build_ignrr_no_noise_aware,
    build_ignrr_no_illum_guidance
)


def evaluate_baseline(dataset, device="cpu"):
    decom = DecomNet().to(device)
    relight = RelightNet().to(device)
    
    decom_path = os.path.join(CORRECTED_BASELINE_DIR, "decom_best.pth")
    relight_path = os.path.join(CORRECTED_BASELINE_DIR, "relight_best.pth")
    
    load_checkpoint(decom, decom_path, device=device)
    load_checkpoint(relight, relight_path, device=device)
    decom.eval()
    relight.eval()
    
    results = []
    times = []
    
    with torch.no_grad():
        for low_t, high_t, name in dataset:
            low_t = low_t.unsqueeze(0).to(device)
            high_t = high_t.unsqueeze(0).to(device)
            
            t0 = time.time()
            R, L = decom(low_t)
            I_delta = relight(L, R)
            enhanced = torch.clamp(R * I_delta.repeat(1, 3, 1, 1), 0.0, 1.0)
            latency = (time.time() - t0) * 1000.0
            times.append(latency)
            
            enh_np = enhanced[0].permute(1, 2, 0).cpu().numpy()
            gt_np = high_t[0].permute(1, 2, 0).cpu().numpy()
            low_np = low_t[0].permute(1, 2, 0).cpu().numpy()
            
            p = compute_psnr(enh_np, gt_np)
            s = compute_ssim(enh_np, gt_np)
            color_m = compute_color_metrics(enh_np, gt_np)
            
            results.append({
                "Image": name,
                "Model": "Baseline Retinex",
                "PSNR": p,
                "SSIM": s,
                "DeltaE2000": color_m["delta_e_2000"],
                "RGB_MAE": color_m["rgb_mae"],
                "L_error": color_m["l_error"],
                "a_error": color_m["a_error"],
                "b_error": color_m["b_error"],
                "Latency_ms": latency
            })
            
    return results, np.mean(times)


def evaluate_ignrr(model, dataset, model_name="IGNRR", device="cpu"):
    model.eval()
    results = []
    times = []
    
    with torch.no_grad():
        for low_t, high_t, name in dataset:
            low_t = low_t.unsqueeze(0).to(device)
            high_t = high_t.unsqueeze(0).to(device)
            
            t0 = time.time()
            outs = model(low_t)
            enhanced = outs["enhanced"]
            latency = (time.time() - t0) * 1000.0
            times.append(latency)
            
            enh_np = enhanced[0].permute(1, 2, 0).cpu().numpy()
            gt_np = high_t[0].permute(1, 2, 0).cpu().numpy()
            
            p = compute_psnr(enh_np, gt_np)
            s = compute_ssim(enh_np, gt_np)
            color_m = compute_color_metrics(enh_np, gt_np)
            
            results.append({
                "Image": name,
                "Model": model_name,
                "PSNR": p,
                "SSIM": s,
                "DeltaE2000": color_m["delta_e_2000"],
                "RGB_MAE": color_m["rgb_mae"],
                "L_error": color_m["l_error"],
                "a_error": color_m["a_error"],
                "b_error": color_m["b_error"],
                "Latency_ms": latency
            })
            
    return results, np.mean(times)


def run_benchmark(device="cpu"):
    print(f"Loading LOL-v1 Test Dataset (15 images)...")
    test_dataset = LOLDataset(DATASET_DIR, split="test", patch_size=None)
    
    all_results = []
    summary_table = []
    
    # 1. Baseline Retinex
    print("\n[1/4] Evaluating Corrected Retinex Baseline...")
    b_res, b_time = evaluate_baseline(test_dataset, device=device)
    all_results.extend(b_res)
    b_df = pd.DataFrame(b_res)
    summary_table.append({
        "Experiment": "A: Baseline Retinex",
        "PSNR (dB)": round(b_df["PSNR"].mean(), 2),
        "SSIM": round(b_df["SSIM"].mean(), 4),
        "Delta E 2000": round(b_df["DeltaE2000"].mean(), 2),
        "RGB MAE": round(b_df["RGB_MAE"].mean(), 4),
        "Latency (ms)": round(b_time, 1)
    })
    
    # 2. Proposed IGNRR
    ignrr_path = os.path.join(IMPROVED_CHECKPOINT_DIR, "ignrr_best.pth")
    if os.path.exists(ignrr_path):
        print("\n[2/4] Evaluating Proposed IGNRR (Full Model)...")
        model = build_ignrr_full().to(device)
        load_checkpoint(model, ignrr_path, device=device)
        i_res, i_time = evaluate_ignrr(model, test_dataset, model_name="IGNRR (Proposed)", device=device)
        all_results.extend(i_res)
        i_df = pd.DataFrame(i_res)
        summary_table.append({
            "Experiment": "B: Baseline + IGNRR (Proposed)",
            "PSNR (dB)": round(i_df["PSNR"].mean(), 2),
            "SSIM": round(i_df["SSIM"].mean(), 4),
            "Delta E 2000": round(i_df["DeltaE2000"].mean(), 2),
            "RGB MAE": round(i_df["RGB_MAE"].mean(), 4),
            "Latency (ms)": round(i_time, 1)
        })
    else:
        print("\n[2/4] Note: IGNRR checkpoint not yet trained. Run training/train_ignrr.py to generate weights.")
        
    # 3. Ablation C (w/o noise awareness)
    c_path = os.path.join(ABLATION_CHECKPOINT_DIR, "no_noise_aware", "no_noise_aware_best.pth")
    if os.path.exists(c_path):
        print("\n[3/4] Evaluating Ablation C (w/o Noise Awareness)...")
        m_c = build_ignrr_no_noise_aware().to(device)
        load_checkpoint(m_c, c_path, device=device)
        c_res, c_time = evaluate_ignrr(m_c, test_dataset, model_name="Ablation: w/o Noise-Aware", device=device)
        all_results.extend(c_res)
        c_df = pd.DataFrame(c_res)
        summary_table.append({
            "Experiment": "C: Refinement w/o Noise Aware",
            "PSNR (dB)": round(c_df["PSNR"].mean(), 2),
            "SSIM": round(c_df["SSIM"].mean(), 4),
            "Delta E 2000": round(c_df["DeltaE2000"].mean(), 2),
            "RGB MAE": round(c_df["RGB_MAE"].mean(), 4),
            "Latency (ms)": round(c_time, 1)
        })

    # 4. Ablation D (w/o illumination guidance)
    d_path = os.path.join(ABLATION_CHECKPOINT_DIR, "no_illum_guidance", "no_illum_guidance_best.pth")
    if os.path.exists(d_path):
        print("\n[4/4] Evaluating Ablation D (w/o Illum Guidance)...")
        m_d = build_ignrr_no_illum_guidance().to(device)
        load_checkpoint(m_d, d_path, device=device)
        d_res, d_time = evaluate_ignrr(m_d, test_dataset, model_name="Ablation: w/o Illum Guidance", device=device)
        all_results.extend(d_res)
        d_df = pd.DataFrame(d_res)
        summary_table.append({
            "Experiment": "D: Refinement w/o Illum Guidance",
            "PSNR (dB)": round(d_df["PSNR"].mean(), 2),
            "SSIM": round(d_df["SSIM"].mean(), 4),
            "Delta E 2000": round(d_df["DeltaE2000"].mean(), 2),
            "RGB MAE": round(d_df["RGB_MAE"].mean(), 4),
            "Latency (ms)": round(d_time, 1)
        })

    summary_df = pd.DataFrame(summary_table)
    print("\n==========================================================================")
    print("                    EVALUATION & BENCHMARK SUMMARY (LOL-v1)")
    print("==========================================================================")
    print(summary_df.to_string(index=False))
    print("==========================================================================\n")
    
    # Save results
    detail_csv = os.path.join(EXPERIMENTS_DIR, "test_benchmark_detailed.csv")
    summary_csv = os.path.join(EXPERIMENTS_DIR, "test_benchmark_summary.csv")
    pd.DataFrame(all_results).to_csv(detail_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)
    print(f"Results saved to:\n  - {summary_csv}\n  - {detail_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()
    run_benchmark(device=args.device)
