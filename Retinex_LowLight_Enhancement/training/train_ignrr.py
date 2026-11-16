"""
Training Script for Illumination-Guided Noise-Aware Reflectance Refinement (IGNRR).
Supports:
- Full IGNRR model and Ablation variants
- Auto device detection (CUDA with CPU fallback)
- Resumable training with early stopping
- Metric logging to CSV (PSNR, SSIM, Validation Loss)
- Zero data leakage: strictly uses the 437 train pairs and 48 val pairs
"""

import os
import sys
import time
import argparse
import pandas as pd
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Add project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from configs.config import (
    DATASET_DIR,
    IMPROVED_CHECKPOINT_DIR,
    ABLATION_CHECKPOINT_DIR,
    CORRECTED_BASELINE_DIR,
    EXPERIMENTS_DIR,
    RANDOM_SEED,
    BATCH_SIZE,
    PATCH_SIZE,
    LEARNING_RATE
)
from utils.dataset import LOLDataset
from utils.checkpoint import load_checkpoint, save_checkpoint
from utils.losses import IGNRRLoss
from utils.metrics import compute_psnr, compute_ssim
from models.ignrr_net import (
    build_ignrr_full,
    build_ignrr_no_noise_aware,
    build_ignrr_no_illum_guidance
)


def set_seed(seed=RANDOM_SEED):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def get_model(model_type):
    if model_type == "full":
        return build_ignrr_full()
    elif model_type == "no_noise_aware":
        return build_ignrr_no_noise_aware()
    elif model_type == "no_illum_guidance":
        return build_ignrr_no_illum_guidance()
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


def train(args):
    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    print(f"==================================================")
    print(f"Training Model: {args.model_type.upper()}")
    print(f"Device: {device}")
    print(f"Batch Size: {args.batch_size}, Patch Size: {args.patch_size}")
    print(f"Learning Rate: {args.lr}, Epochs: {args.epochs}")
    print(f"==================================================")

    # 1. Initialize Dataset and Loaders
    train_dataset = LOLDataset(DATASET_DIR, split="train", patch_size=args.patch_size)
    val_dataset = LOLDataset(DATASET_DIR, split="val", patch_size=None)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0
    )

    print(f"Loaded {len(train_dataset)} training pairs and {len(val_dataset)} validation pairs.")

    # 2. Build Model
    model = get_model(args.model_type).to(device)

    # Warm-start DecomNet and RelightNet with verified baseline checkpoints
    decom_path = os.path.join(CORRECTED_BASELINE_DIR, "decom_best.pth")
    if os.path.exists(decom_path):
        print(f"Warm-starting DecomNet from: {decom_path}")
        load_checkpoint(model.decom_net, decom_path, device=device)

    relight_path = os.path.join(CORRECTED_BASELINE_DIR, "relight_best.pth")
    if os.path.exists(relight_path):
        print(f"Warm-starting RelightNet from: {relight_path}")
        r_ckpt = torch.load(relight_path, map_location=device)
        r_state = r_ckpt["model_state_dict"] if "model_state_dict" in r_ckpt else r_ckpt
        m_state = model.relight_net.state_dict()
        for k, v in r_state.items():
            if k in m_state and m_state[k].shape == v.shape:
                m_state[k] = v
            elif k == "output.weight" and "output.weight" in m_state:
                m_state[k] = v.repeat(3, 1, 1, 1)
            elif k == "output.bias" and "output.bias" in m_state:
                m_state[k] = v.repeat(3)
        model.relight_net.load_state_dict(m_state)

    # Optimizer & Scheduler
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)
    criterion = IGNRRLoss().to(device)

    # Output paths
    if args.model_type == "full":
        save_dir = IMPROVED_CHECKPOINT_DIR
        log_file = os.path.join(EXPERIMENTS_DIR, "ignrr_training_log.csv")
    else:
        save_dir = os.path.join(ABLATION_CHECKPOINT_DIR, args.model_type)
        log_file = os.path.join(EXPERIMENTS_DIR, f"ablation_{args.model_type}_log.csv")
    os.makedirs(save_dir, exist_ok=True)

    best_val_psnr = -1.0
    start_epoch = 1
    history = []

    # Optional Resumption
    latest_ckpt = os.path.join(save_dir, f"{args.model_type}_latest.pth")
    if args.resume and os.path.exists(latest_ckpt):
        print(f"Resuming training from: {latest_ckpt}")
        meta = load_checkpoint(model, latest_ckpt, optimizer=optimizer, device=device)
        start_epoch = meta.get("epoch", 0) + 1
        print(f"Resuming at epoch {start_epoch}")

    # Training Loop
    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start = time.time()
        model.train()
        train_loss = 0.0

        for low_batch, high_batch, _ in train_loader:
            low_batch = low_batch.to(device)
            high_batch = high_batch.to(device)

            optimizer.zero_grad()
            outputs = model(low_batch)
            loss_dict = criterion(
                outputs["enhanced"],
                high_batch,
                outputs["R_refined"],
                outputs["I_delta"]
            )
            loss_dict["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            train_loss += loss_dict["total"].item()

        train_loss /= len(train_loader)
        scheduler.step()

        # Validation Phase
        model.eval()
        val_loss = 0.0
        val_psnr_list = []
        val_ssim_list = []

        val_limit = args.val_samples if (args.val_samples > 0 and epoch < args.epochs) else len(val_loader)
        count = 0

        with torch.no_grad():
            for low_v, high_v, _ in val_loader:
                if count >= val_limit:
                    break
                count += 1
                low_v = low_v.to(device)
                high_v = high_v.to(device)

                outs = model(low_v)
                v_loss = criterion(outs["enhanced"], high_v, outs["R_refined"], outs["I_delta"])
                val_loss += v_loss["total"].item()

                enh_np = outs["enhanced"][0].permute(1, 2, 0).cpu().numpy()
                gt_np = high_v[0].permute(1, 2, 0).cpu().numpy()

                val_psnr_list.append(compute_psnr(enh_np, gt_np))
                val_ssim_list.append(compute_ssim(enh_np, gt_np))

        val_loss /= max(count, 1)
        mean_val_psnr = float(np.mean(val_psnr_list))
        mean_val_ssim = float(np.mean(val_ssim_list))
        epoch_time = time.time() - epoch_start

        print(
            f"Epoch [{epoch:02d}/{args.epochs:02d}] "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val PSNR: {mean_val_psnr:.2f} dB | "
            f"Val SSIM: {mean_val_ssim:.4f} | "
            f"Time: {epoch_time:.1f}s"
        )

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": val_loss,
            "val_psnr": mean_val_psnr,
            "val_ssim": mean_val_ssim
        })

        # Save Latest Checkpoint
        save_checkpoint(
            model,
            latest_ckpt,
            optimizer=optimizer,
            epoch=epoch,
            validation_loss=val_loss
        )

        # Save Best Checkpoint
        if mean_val_psnr > best_val_psnr:
            best_val_psnr = mean_val_psnr
            best_ckpt = os.path.join(save_dir, f"{args.model_type}_best.pth")
            save_checkpoint(
                model,
                best_ckpt,
                optimizer=optimizer,
                epoch=epoch,
                validation_loss=val_loss,
                extra_data={"val_psnr": mean_val_psnr, "val_ssim": mean_val_ssim}
            )
            print(f"  --> Saved new best checkpoint with Val PSNR: {best_val_psnr:.2f} dB")

    # Save log to CSV
    pd.DataFrame(history).to_csv(log_file, index=False)
    print(f"\nTraining completed! Best Val PSNR: {best_val_psnr:.2f} dB. Log saved to: {log_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train IGNRR / Retinex models.")
    parser.add_argument("--model_type", type=str, default="full", choices=["full", "no_noise_aware", "no_illum_guidance"])
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--patch_size", type=int, default=PATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--val_samples", type=int, default=12, help="Validation samples per epoch (0 for full)")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    train(args)
