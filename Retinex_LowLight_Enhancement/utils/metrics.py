"""
Comprehensive Evaluation Metrics for Low-Light Image Enhancement.
Includes:
- PSNR (Peak Signal-to-Noise Ratio)
- SSIM (Structural Similarity Index, Wang et al. 2004)
- Delta E 2000 (CIEDE2000 Perceptual Color Difference)
- RGB MAE (Mean Absolute Error across color channels)
- CIELAB Channel Errors (L*, a*, b* errors)
- Edge Gradient Error (Sobel gradient difference)
"""

import math
import numpy as np
import torch
import torch.nn.functional as F


# ====================================================================
# PyTorch & NumPy PSNR
# ====================================================================

def compute_psnr(img1, img2, data_range=1.0):
    """
    Computes PSNR between two images (numpy arrays in [0, 1]).
    """
    if isinstance(img1, torch.Tensor):
        img1 = img1.detach().cpu().numpy()
    if isinstance(img2, torch.Tensor):
        img2 = img2.detach().cpu().numpy()
        
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float("inf")
    return 20.0 * math.log10(data_range) - 10.0 * math.log10(mse)


# ====================================================================
# PyTorch & NumPy SSIM (Wang et al. 2004 with 11x11 Gaussian window)
# ====================================================================

def _gaussian(window_size, sigma):
    gauss = torch.Tensor([
        math.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2))
        for x in range(window_size)
    ])
    return gauss / gauss.sum()


def _create_window(window_size, channel=3):
    _1D_window = _gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


def compute_ssim_torch(img1, img2, window_size=11):
    """
    img1, img2: torch.Tensor of shape (B, C, H, W) in [0, 1]
    """
    channel = img1.size(1)
    window = _create_window(window_size, channel).to(img1.device)
    
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)
    
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2
    
    c1 = (0.01) ** 2
    c2 = (0.03) ** 2
    
    ssim_map = ((2 * mu1_mu2 + c1) * (2 * sigma12 + c2)) / ((mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2))
    return ssim_map.mean().item()


def compute_ssim(img1, img2):
    """
    Numpy wrapper for SSIM: (H, W, 3) or (H, W) in [0, 1].
    """
    if img1.ndim == 2:
        img1 = np.expand_dims(img1, axis=2)
        img2 = np.expand_dims(img2, axis=2)
    t1 = torch.from_numpy(img1.transpose(2, 0, 1)).unsqueeze(0).float()
    t2 = torch.from_numpy(img2.transpose(2, 0, 1)).unsqueeze(0).float()
    return compute_ssim_torch(t1, t2)


# ====================================================================
# Color Space Conversion & Delta E 2000 (CIEDE2000)
# ====================================================================

def rgb_to_xyz(rgb):
    """
    Converts sRGB [0, 1] to CIE XYZ using sRGB D65 standard illuminant.
    """
    # Gamma expansion
    mask = rgb > 0.04045
    rgb_lin = np.empty_like(rgb)
    rgb_lin[mask] = ((rgb[mask] + 0.055) / 1.055) ** 2.4
    rgb_lin[~mask] = rgb[~mask] / 12.92
    
    # Transformation matrix sRGB to XYZ (D65)
    r = rgb_lin[:, :, 0]
    g = rgb_lin[:, :, 1]
    b = rgb_lin[:, :, 2]
    
    x = r * 0.4124564 + g * 0.3575761 + b * 0.1804375
    y = r * 0.2126729 + g * 0.7151522 + b * 0.0721750
    z = r * 0.0193339 + g * 0.1191920 + b * 0.9503041
    
    return np.stack([x, y, z], axis=2)


def xyz_to_lab(xyz):
    """
    Converts CIE XYZ to CIE L*a*b* under D65 reference white.
    """
    # D65 reference white
    xn, yn, zn = 0.95047, 1.00000, 1.08883
    
    x = xyz[:, :, 0] / xn
    y = xyz[:, :, 1] / yn
    z = xyz[:, :, 2] / zn
    
    delta = 6.0 / 29.0
    
    def f(t):
        mask = t > (delta ** 3)
        res = np.empty_like(t)
        res[mask] = t[mask] ** (1.0 / 3.0)
        res[~mask] = t[~mask] / (3.0 * delta ** 2) + 4.0 / 29.0
        return res
    
    fx = f(x)
    fy = f(y)
    fz = f(z)
    
    l = 116.0 * fy - 16.0
    a = 500.0 * (fx - fy)
    b = 200.0 * (fy - fz)
    
    return np.stack([l, a, b], axis=2)


def rgb_to_lab(rgb):
    return xyz_to_lab(rgb_to_xyz(rgb))


def compute_delta_e_2000(lab1, lab2):
    """
    Computes CIEDE2000 color difference between two LAB images.
    Returns the mean Delta E 2000 scalar.
    """
    l1, a1, b1 = lab1[:, :, 0], lab1[:, :, 1], lab1[:, :, 2]
    l2, a2, b2 = lab2[:, :, 0], lab2[:, :, 1], lab2[:, :, 2]
    
    avg_l = (l1 + l2) / 2.0
    c1 = np.sqrt(a1**2 + b1**2)
    c2 = np.sqrt(a2**2 + b2**2)
    avg_c = (c1 + c2) / 2.0
    
    g = 0.5 * (1.0 - np.sqrt((avg_c ** 7) / ((avg_c ** 7) + 25.0 ** 7 + 1e-12)))
    a1_prime = (1.0 + g) * a1
    a2_prime = (1.0 + g) * a2
    
    c1_prime = np.sqrt(a1_prime**2 + b1**2)
    c2_prime = np.sqrt(a2_prime**2 + b2**2)
    avg_c_prime = (c1_prime + c2_prime) / 2.0
    
    h1_prime = np.degrees(np.arctan2(b1, a1_prime)) % 360.0
    h2_prime = np.degrees(np.arctan2(b2, a2_prime)) % 360.0
    
    # Delta h prime
    diff_h = h2_prime - h1_prime
    abs_diff = np.abs(diff_h)
    delta_h_prime = np.where(
        abs_diff <= 180.0,
        diff_h,
        np.where(h2_prime <= h1_prime, diff_h + 360.0, diff_h - 360.0)
    )
    delta_h_prime = np.where((c1_prime == 0) | (c2_prime == 0), 0.0, delta_h_prime)
    
    delta_l_prime = l2 - l1
    delta_c_prime = c2_prime - c1_prime
    delta_capital_h_prime = 2.0 * np.sqrt(c1_prime * c2_prime + 1e-12) * np.sin(np.radians(delta_h_prime / 2.0))
    
    avg_l_prime = (l1 + l2) / 2.0
    
    sum_h = h1_prime + h2_prime
    avg_h_prime = np.where(
        abs_diff > 180.0,
        np.where(sum_h < 360.0, (sum_h + 360.0) / 2.0, (sum_h - 360.0) / 2.0),
        sum_h / 2.0
    )
    avg_h_prime = np.where((c1_prime == 0) | (c2_prime == 0), sum_h, avg_h_prime)
    
    t = (1.0 
         - 0.17 * np.cos(np.radians(avg_h_prime - 30.0))
         + 0.24 * np.cos(np.radians(2.0 * avg_h_prime))
         + 0.32 * np.cos(np.radians(3.0 * avg_h_prime + 6.0))
         - 0.20 * np.cos(np.radians(4.0 * avg_h_prime - 63.0)))
         
    sl = 1.0 + (0.015 * ((avg_l_prime - 50.0) ** 2)) / np.sqrt(20.0 + (avg_l_prime - 50.0) ** 2 + 1e-12)
    sc = 1.0 + 0.045 * avg_c_prime
    sh = 1.0 + 0.015 * avg_c_prime * t
    
    delta_theta = 30.0 * np.exp(-(((avg_h_prime - 275.0) / 25.0) ** 2))
    rc = 2.0 * np.sqrt((avg_c_prime ** 7) / ((avg_c_prime ** 7) + 25.0 ** 7 + 1e-12))
    rt = -np.sin(np.radians(2.0 * delta_theta)) * rc
    
    kl, kc, kh = 1.0, 1.0, 1.0
    delta_e = np.sqrt(
        (delta_l_prime / (kl * sl)) ** 2
        + (delta_c_prime / (kc * sc)) ** 2
        + (delta_capital_h_prime / (kh * sh)) ** 2
        + rt * (delta_c_prime / (kc * sc)) * (delta_capital_h_prime / (kh * sh))
        + 1e-12
    )
    return float(np.mean(delta_e))


def compute_color_metrics(enhanced_rgb, ground_truth_rgb):
    """
    Computes all perceptual color metrics between enhanced and ground truth RGB images:
    - Delta E 2000
    - RGB MAE
    - Mean L* error, a* error, b* error
    """
    rgb_mae = float(np.mean(np.abs(enhanced_rgb - ground_truth_rgb)))
    
    lab_enh = rgb_to_lab(np.clip(enhanced_rgb, 0.0, 1.0))
    lab_gt = rgb_to_lab(np.clip(ground_truth_rgb, 0.0, 1.0))
    
    delta_e = compute_delta_e_2000(lab_enh, lab_gt)
    
    l_error = float(np.mean(np.abs(lab_enh[:, :, 0] - lab_gt[:, :, 0])))
    a_error = float(np.mean(np.abs(lab_enh[:, :, 1] - lab_gt[:, :, 1])))
    b_error = float(np.mean(np.abs(lab_enh[:, :, 2] - lab_gt[:, :, 2])))
    
    return {
        "delta_e_2000": delta_e,
        "rgb_mae": rgb_mae,
        "l_error": l_error,
        "a_error": a_error,
        "b_error": b_error
    }
