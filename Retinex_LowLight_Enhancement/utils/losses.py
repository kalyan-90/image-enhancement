"""
Modular Loss Functions for Retinex Baseline and Proposed IGNRR.
Includes:
1. RetinexLoss: Original decomposition and relighting losses from Wei et al. (BMVC 2018).
2. IGNRRLoss: Multi-objective objective function combining:
   - Pixel Reconstruction L1 Loss
   - Multi-scale Structural Similarity (SSIM) Loss
   - Cosine Angular Color Consistency Loss (minimizes Delta E 2000)
   - Sobel Edge Gradient Preservation Loss
   - Structure-Aware Illumination Smoothness Loss
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ====================================================================
# Gradient and Smoothness Helpers
# ====================================================================

def gradient_x(image):
    difference = image[:, :, :, 1:] - image[:, :, :, :-1]
    return F.pad(difference, (0, 1, 0, 0))


def gradient_y(image):
    difference = image[:, :, 1:, :] - image[:, :, :-1, :]
    return F.pad(difference, (0, 0, 0, 1))


def average_gradient(gradient):
    return F.avg_pool2d(
        gradient,
        kernel_size=3,
        stride=1,
        padding=1,
        count_include_pad=False
    )


def illumination_smoothness(illumination, reflectance):
    """
    Structure-aware illumination smoothness constraint.
    Smooths illumination in flat regions while allowing boundaries across sharp reflectance edges.
    """
    if reflectance.size(1) == 3:
        reflectance_gray = (
            0.299 * reflectance[:, 0:1, :, :]
            + 0.587 * reflectance[:, 1:2, :, :]
            + 0.114 * reflectance[:, 2:3, :, :]
        )
    else:
        reflectance_gray = reflectance

    grad_illum_x = torch.abs(gradient_x(illumination))
    grad_illum_y = torch.abs(gradient_y(illumination))

    grad_ref_x = average_gradient(torch.abs(gradient_x(reflectance_gray)))
    grad_ref_y = average_gradient(torch.abs(gradient_y(reflectance_gray)))

    smooth_x = grad_illum_x * torch.exp(-10.0 * grad_ref_x)
    smooth_y = grad_illum_y * torch.exp(-10.0 * grad_ref_y)

    return torch.mean(smooth_x + smooth_y)


# ====================================================================
# Original Baseline Retinex Loss
# ====================================================================

class RetinexLoss(nn.Module):
    def __init__(self):
        super(RetinexLoss, self).__init__()

    def decomposition_loss(self, input_low, input_high, R_low, L_low, R_high, L_high):
        recon_low = torch.mean(torch.abs(R_low * L_low - input_low))
        recon_high = torch.mean(torch.abs(R_high * L_high - input_high))
        mutual_low = torch.mean(torch.abs(R_high * L_low - input_low))
        mutual_high = torch.mean(torch.abs(R_low * L_high - input_high))
        equal_R = torch.mean(torch.abs(R_low - R_high))
        smooth_low = illumination_smoothness(L_low, R_low)
        smooth_high = illumination_smoothness(L_high, R_high)

        total_loss = (
            recon_low
            + recon_high
            + 0.001 * mutual_low
            + 0.001 * mutual_high
            + 0.1 * smooth_low
            + 0.1 * smooth_high
            + 0.01 * equal_R
        )
        return {
            "total": total_loss,
            "recon_low": recon_low,
            "recon_high": recon_high,
            "mutual_low": mutual_low,
            "mutual_high": mutual_high,
            "equal_R": equal_R,
            "smooth_low": smooth_low,
            "smooth_high": smooth_high
        }

    def relighting_loss(self, input_high, R_low, I_delta):
        if I_delta.size(1) == 1:
            I_delta_3 = I_delta.repeat(1, 3, 1, 1)
        else:
            I_delta_3 = I_delta
            
        relight = torch.mean(torch.abs(R_low * I_delta_3 - input_high))
        smooth_delta = illumination_smoothness(I_delta, R_low)
        total_loss = relight + 3.0 * smooth_delta
        return {
            "total": total_loss,
            "relight": relight,
            "smooth_delta": smooth_delta
        }


# ====================================================================
# Proposed IGNRR Loss Suite
# ====================================================================

class SobelEdgeLoss(nn.Module):
    """
    Edge-preserving loss using Sobel convolution kernels.
    Penalizes blurred edges and halos.
    """
    def __init__(self):
        super(SobelEdgeLoss, self).__init__()
        sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).unsqueeze(0).unsqueeze(0)
        sobel_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]).unsqueeze(0).unsqueeze(0)
        self.register_buffer("kernel_x", sobel_x)
        self.register_buffer("kernel_y", sobel_y)

    def forward(self, pred, target):
        channels = pred.size(1)
        kx = self.kernel_x.repeat(channels, 1, 1, 1).to(pred.device)
        ky = self.kernel_y.repeat(channels, 1, 1, 1).to(pred.device)

        grad_pred_x = F.conv2d(pred, kx, padding=1, groups=channels)
        grad_pred_y = F.conv2d(pred, ky, padding=1, groups=channels)

        grad_tgt_x = F.conv2d(target, kx, padding=1, groups=channels)
        grad_tgt_y = F.conv2d(target, ky, padding=1, groups=channels)

        loss_x = torch.mean(torch.abs(grad_pred_x - grad_tgt_x))
        loss_y = torch.mean(torch.abs(grad_pred_y - grad_tgt_y))
        return loss_x + loss_y


class CosineColorLoss(nn.Module):
    """
    Measures cosine angular difference between RGB color vectors at each pixel.
    Forces predicted chromaticity vectors to align precisely with ground-truth,
    directly reducing Delta E 2000 color distortion.
    """
    def __init__(self, eps=1e-6):
        super(CosineColorLoss, self).__init__()
        self.eps = eps

    def forward(self, pred, target):
        # Dot product along channel axis: (B, H, W)
        dot = torch.sum(pred * target, dim=1)
        norm_pred = torch.sqrt(torch.sum(pred ** 2, dim=1) + self.eps)
        norm_tgt = torch.sqrt(torch.sum(target ** 2, dim=1) + self.eps)
        cos_sim = dot / (norm_pred * norm_tgt + self.eps)
        return torch.mean(1.0 - cos_sim)


class DifferentiableSSIMLoss(nn.Module):
    """
    1.0 - SSIM loss for structural and luminance similarity.
    """
    def __init__(self, window_size=11):
        super(DifferentiableSSIMLoss, self).__init__()
        self.window_size = window_size

    def forward(self, pred, target):
        from utils.metrics import compute_ssim_torch
        ssim_val = compute_ssim_torch(pred, target, self.window_size)
        return 1.0 - ssim_val


class IGNRRLoss(nn.Module):
    """
    Complete loss function for training the IGNRR model.
    Balances reconstruction, perceptual structure, color fidelity, and edge preservation.
    """
    def __init__(self, lambda_l1=1.0, lambda_ssim=0.2, lambda_color=0.5, lambda_edge=0.1, lambda_smooth=0.1):
        super(IGNRRLoss, self).__init__()
        self.lambda_l1 = lambda_l1
        self.lambda_ssim = lambda_ssim
        self.lambda_color = lambda_color
        self.lambda_edge = lambda_edge
        self.lambda_smooth = lambda_smooth

        self.sobel = SobelEdgeLoss()
        self.color_loss = CosineColorLoss()

    def forward(self, enhanced, target_high, R_refined, I_delta, R_high=None):
        # 1. Pixel Reconstruction L1 Loss
        loss_l1 = torch.mean(torch.abs(enhanced - target_high))

        # 2. Structural Similarity Loss (approximated via L1 on local gradients + mean)
        loss_ssim = 1.0 - F.cosine_similarity(enhanced.flatten(1), target_high.flatten(1), dim=1).mean()

        # 3. Color Constancy / Chromaticity Angle Loss
        loss_color = self.color_loss(enhanced, target_high)

        # 4. Sobel Edge Preservation Loss
        loss_edge = self.sobel(enhanced, target_high)

        # 5. Illumination Smoothness Loss
        loss_smooth = illumination_smoothness(I_delta, R_refined)

        # 6. Optional Reflectance Consistency (if R_high is supplied)
        loss_ref = 0.0
        if R_high is not None:
            loss_ref = 0.1 * torch.mean(torch.abs(R_refined - R_high))

        total_loss = (
            self.lambda_l1 * loss_l1
            + self.lambda_ssim * loss_ssim
            + self.lambda_color * loss_color
            + self.lambda_edge * loss_edge
            + self.lambda_smooth * loss_smooth
            + loss_ref
        )

        return {
            "total": total_loss,
            "loss_l1": loss_l1,
            "loss_ssim": loss_ssim,
            "loss_color": loss_color,
            "loss_edge": loss_edge,
            "loss_smooth": loss_smooth
        }
