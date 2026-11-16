"""
Visualization and Comparative Analysis Script.
Generates publication-quality visual comparison figures showing:
[Low-Light Input] | [Ground Truth] | [Baseline Output] | [IGNRR Output] | [Guidance / Illumination Map]
Saves side-by-side comparison images and a multi-image montage grid for the research report.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from configs.config import (
    DATASET_DIR,
    CORRECTED_BASELINE_DIR,
    IMPROVED_CHECKPOINT_DIR,
    RESULTS_DIR
)
from utils.checkpoint import load_checkpoint
from utils.metrics import compute_psnr, compute_ssim, compute_color_metrics
from models.decom_net import DecomNet
from models.relight_net import RelightNet
from models.ignrr_net import build_ignrr_full


def generate_visual_comparisons(device="cpu", max_samples=5):
    test_low_dir = os.path.join(DATASET_DIR, "test", "low")
    test_high_dir = os.path.join(DATASET_DIR, "test", "high")
    comp_dir = os.path.join(RESULTS_DIR, "comparisons")
    os.makedirs(comp_dir, exist_ok=True)

    # 1. Load Baseline
    decom = DecomNet().to(device)
    relight = RelightNet().to(device)
    load_checkpoint(decom, os.path.join(CORRECTED_BASELINE_DIR, "decom_best.pth"), device=device)
    load_checkpoint(relight, os.path.join(CORRECTED_BASELINE_DIR, "relight_best.pth"), device=device)
    decom.eval()
    relight.eval()

    # 2. Load IGNRR (if available)
    ignrr_model = None
    ignrr_path = os.path.join(IMPROVED_CHECKPOINT_DIR, "ignrr_best.pth")
    if os.path.exists(ignrr_path):
        ignrr_model = build_ignrr_full().to(device)
        load_checkpoint(ignrr_model, ignrr_path, device=device)
        ignrr_model.eval()

    test_files = sorted(os.listdir(test_low_dir))[:max_samples]

    for fname in test_files:
        low_path = os.path.join(test_low_dir, fname)
        high_path = os.path.join(test_high_dir, fname)

        low_im = Image.open(low_path).convert("RGB")
        high_im = Image.open(high_path).convert("RGB")

        low_np = np.asarray(low_im, dtype=np.float32) / 255.0
        high_np = np.asarray(high_im, dtype=np.float32) / 255.0

        low_t = torch.from_numpy(low_np).permute(2, 0, 1).unsqueeze(0).to(device)

        # Baseline inference
        with torch.no_grad():
            R_base, L_base = decom(low_t)
            I_delta_base = relight(L_base, R_base)
            base_enh = torch.clamp(R_base * I_delta_base.repeat(1, 3, 1, 1), 0.0, 1.0)
            base_np = base_enh[0].permute(1, 2, 0).cpu().numpy()

        p_base = compute_psnr(base_np, high_np)
        s_base = compute_ssim(base_np, high_np)
        c_base = compute_color_metrics(base_np, high_np)

        # IGNRR inference
        ignrr_np = None
        guidance_np = None
        if ignrr_model is not None:
            with torch.no_grad():
                outs = ignrr_model(low_t)
                ignrr_enh = outs["enhanced"]
                ignrr_np = ignrr_enh[0].permute(1, 2, 0).cpu().numpy()
                guidance_np = (1.0 - outs["L_low"][0, 0]).cpu().numpy()

            p_ignrr = compute_psnr(ignrr_np, high_np)
            s_ignrr = compute_ssim(ignrr_np, high_np)
            c_ignrr = compute_color_metrics(ignrr_np, high_np)

        # Plot comparison figure
        cols = 5 if ignrr_model is not None else 3
        fig, axes = plt.subplots(1, cols, figsize=(cols * 4, 4), dpi=150)

        axes[0].imshow(low_np)
        axes[0].set_title(f"Low-Light Input\n({fname})", fontsize=10)
        axes[0].axis("off")

        axes[1].imshow(high_np)
        axes[1].set_title(f"Ground Truth Reference\n(Normal Light)", fontsize=10)
        axes[1].axis("off")

        axes[2].imshow(base_np)
        axes[2].set_title(
            f"Baseline RetinexNet\nPSNR: {p_base:.2f}dB | SSIM: {s_base:.3f}\nΔE: {c_base['delta_e_2000']:.1f}",
            fontsize=9
        )
        axes[2].axis("off")

        if ignrr_model is not None:
            axes[3].imshow(ignrr_np)
            axes[3].set_title(
                f"Proposed IGNRR\nPSNR: {p_ignrr:.2f}dB | SSIM: {s_ignrr:.3f}\nΔE: {c_ignrr['delta_e_2000']:.1f}",
                fontsize=9
            )
            axes[3].axis("off")

            im_guid = axes[4].imshow(guidance_np, cmap="inferno")
            axes[4].set_title("Illum. Guidance Map\nG = (1 - L_low)", fontsize=9)
            axes[4].axis("off")
            plt.colorbar(im_guid, ax=axes[4], fraction=0.046, pad=0.04)

        plt.tight_layout()
        save_path = os.path.join(comp_dir, f"comparison_{os.path.splitext(fname)[0]}.png")
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
        print(f"Generated comparison: {save_path}")

    print(f"\nAll visual comparisons saved in: {comp_dir}")


if __name__ == "__main__":
    generate_visual_comparisons()
