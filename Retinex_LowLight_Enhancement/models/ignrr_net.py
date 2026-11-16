"""
Full Architecture for Illumination-Guided Noise-Aware Reflectance Refinement (IGNRR).
Combines:
1. DecomNet: Decomposes low-light image into noisy reflectance R_low and illumination L_low.
2. NoiseAwareReflectanceRefiner (NARR): Refines R_low conditioned on L_low and spatial guidance G.
3. ChromaticityRelightNet: Predicts 3-channel relit illumination I_delta.
4. Final Enhancement: I_enh = R_refined * I_delta.
"""

import torch
import torch.nn as nn

from models.decom_net import DecomNet
from models.relight_net import RelightNet
from models.modules import NoiseAwareReflectanceRefiner, ChromaticityRelightNet


class IGNRRNet(nn.Module):
    """
    Proposed Novel Architecture:
    Deep Retinex Network with Illumination-Guided Noise-Aware Reflectance Refinement (IGNRR).
    """
    def __init__(self, use_noise_aware=True, use_illum_guidance=True, use_chromaticity_relight=True):
        super(IGNRRNet, self).__init__()
        
        self.decom_net = DecomNet()
        self.refine_net = NoiseAwareReflectanceRefiner(
            feat_channels=64,
            num_blocks=4,
            use_noise_aware=use_noise_aware,
            use_illum_guidance=use_illum_guidance
        )
        
        self.use_chromaticity_relight = use_chromaticity_relight
        if use_chromaticity_relight:
            self.relight_net = ChromaticityRelightNet()
        else:
            self.relight_net = RelightNet()

    def forward(self, input_low):
        """
        Forward pass of IGNRR.
        Args:
            input_low: Tensor of shape (B, 3, H, W) in [0, 1].
        Returns:
            Dictionary containing:
                - R_low: Decomposed raw reflectance (B, 3, H, W)
                - L_low: Decomposed low-light illumination (B, 1, H, W)
                - R_refined: Denoised, edge-preserved reflectance (B, 3, H, W)
                - I_delta: Relit illumination map (B, 3, H, W) or (B, 1, H, W)
                - enhanced: Final enhanced image (B, 3, H, W)
        """
        # Step 1: Decomposition
        R_low, L_low = self.decom_net(input_low)
        
        # Step 2: Illumination-Guided Noise-Aware Reflectance Refinement
        R_refined = self.refine_net(R_low, L_low)
        
        # Step 3: Chromaticity-Preserving Relighting
        I_delta = self.relight_net(L_low, R_refined)
        
        # Step 4: Final Recombination
        if I_delta.size(1) == 1:
            I_delta_3 = I_delta.repeat(1, 3, 1, 1)
        else:
            I_delta_3 = I_delta
            
        enhanced = torch.clamp(R_refined * I_delta_3, 0.0, 1.0)
        
        return {
            "R_low": R_low,
            "L_low": L_low,
            "R_refined": R_refined,
            "I_delta": I_delta,
            "enhanced": enhanced
        }


# Ablation Model Helpers
def build_ignrr_full():
    """Experiment B: Baseline + Full Proposed IGNRR Refinement."""
    return IGNRRNet(use_noise_aware=True, use_illum_guidance=True, use_chromaticity_relight=True)

def build_ignrr_no_noise_aware():
    """Experiment C: Baseline + Refinement without noise-aware component."""
    return IGNRRNet(use_noise_aware=False, use_illum_guidance=True, use_chromaticity_relight=True)

def build_ignrr_no_illum_guidance():
    """Experiment D: Baseline + Refinement without illumination guidance."""
    return IGNRRNet(use_noise_aware=True, use_illum_guidance=False, use_chromaticity_relight=True)
