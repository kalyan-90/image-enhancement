"""
Architectural Modules for Illumination-Guided Noise-Aware Reflectance Refinement (IGNRR).
Contains:
1. IlluminationGuidanceModule (IGM): Computes spatially adaptive noise risk priors.
2. SpatialAndChannelAttention: Dual attention to isolate high-noise spatial regions and color channels.
3. NoiseAwareRefinementBlock (NARB): Residual refinement block conditioned on illumination.
4. NoiseAwareReflectanceRefiner (NARR): Full multi-scale reflectance denoising sub-network.
5. ChromaticityRelightNet: Chromaticity-preserving multi-channel illumination enhancement.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class IlluminationGuidanceModule(nn.Module):
    """
    Computes spatially adaptive noise guidance map G from estimated illumination L.
    Mathematical Motivation:
        In Retinex theory, I = R * L + n => R = I/L - n/L.
        Noise variance in reflectance scales as sigma_R^2 proportional to 1 / (L^2 + eps).
        Dark regions (L -> 0) have extreme noise amplification; bright regions (L -> 1) have minimal noise.
        Guidance prior G(x, y) = 1.0 - L(x, y) acts as a spatial attention mask:
        G in [0, 1], where 1 represents high noise risk, and 0 represents clean signal.
    """
    def __init__(self):
        super(IlluminationGuidanceModule, self).__init__()

    def forward(self, L):
        """
        Args:
            L: Illumination map tensor of shape (B, 1, H, W) in [0, 1].
        Returns:
            G: Spatial noise risk guidance map of shape (B, 1, H, W) in [0, 1].
        """
        return torch.clamp(1.0 - L, 0.0, 1.0)


class SpatialAndChannelAttention(nn.Module):
    """
    Dual attention mechanism:
    - Channel attention captures inter-channel color correlations (R, G, B sensor gains).
    - Spatial attention focuses refinement specifically on shadow regions.
    """
    def __init__(self, channels, reduction=8):
        super(SpatialAndChannelAttention, self).__init__()
        
        # Channel Attention
        self.fc1 = nn.Conv2d(channels, max(channels // reduction, 8), 1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(max(channels // reduction, 8), channels, 1, bias=False)
        self.sigmoid_ch = nn.Sigmoid()
        
        # Spatial Attention
        self.spatial_conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid_sp = nn.Sigmoid()

    def forward(self, x):
        # Channel attention
        avg_pool = torch.mean(x, dim=(2, 3), keepdim=True)
        ch_attn = self.sigmoid_ch(self.fc2(self.relu(self.fc1(avg_pool))))
        out = x * ch_attn
        
        # Spatial attention
        avg_out = torch.mean(out, dim=1, keepdim=True)
        max_out = torch.max(out, dim=1, keepdim=True)[0]
        sp_in = torch.cat([avg_out, max_out], dim=1)
        sp_attn = self.sigmoid_sp(self.spatial_conv(sp_in))
        
        return out * sp_attn


class NoiseAwareRefinementBlock(nn.Module):
    """
    Residual convolutional block with dilated receptive field and attention.
    Extracts multi-scale noise features without destroying sharp edges.
    """
    def __init__(self, channels=64, dilation=1):
        super(NoiseAwareRefinementBlock, self).__init__()
        padding = dilation
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=padding, dilation=dilation, bias=False)
        self.norm1 = nn.InstanceNorm2d(channels, affine=True)
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.norm2 = nn.InstanceNorm2d(channels, affine=True)
        self.attention = SpatialAndChannelAttention(channels)

    def forward(self, x):
        residual = x
        out = self.relu(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        out = self.attention(out)
        return self.relu(residual + out)


class NoiseAwareReflectanceRefiner(nn.Module):
    """
    Refines noisy reflectance R_low using illumination guidance G = (1 - L_low).
    Inputs:
        R_low: (B, 3, H, W)
        L_low: (B, 1, H, W)
    Output:
        R_refined: (B, 3, H, W)
    """
    def __init__(self, in_channels=5, feat_channels=64, num_blocks=4, use_noise_aware=True, use_illum_guidance=True):
        super(NoiseAwareReflectanceRefiner, self).__init__()
        self.use_noise_aware = use_noise_aware
        self.use_illum_guidance = use_illum_guidance
        self.guidance = IlluminationGuidanceModule()
        
        actual_in = 3
        if self.use_illum_guidance:
            actual_in += 2  # L_low (1) + G (1)
            
        self.head = nn.Sequential(
            nn.Conv2d(actual_in, feat_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        dilations = [1, 2, 3, 1] if self.use_noise_aware else [1, 1, 1, 1]
        self.blocks = nn.ModuleList([
            NoiseAwareRefinementBlock(feat_channels, dilation=d)
            for d in dilations[:num_blocks]
        ])
        
        self.tail = nn.Sequential(
            nn.Conv2d(feat_channels, 3, kernel_size=3, padding=1)
        )
        # Zero-initialize tail so initial refinement is identity: R_refined = R_low
        nn.init.zeros_(self.tail[0].weight)
        nn.init.zeros_(self.tail[0].bias)

    def forward(self, R_low, L_low):
        if not self.use_illum_guidance:
            # Ablation D: Without illumination guidance
            inp = R_low
            G = torch.ones_like(L_low)
        else:
            G = self.guidance(L_low)
            inp = torch.cat([R_low, L_low, G], dim=1)
            
        feats = self.head(inp)
        for block in self.blocks:
            feats = block(feats)
        
        noise_residual = self.tail(feats)
        
        if self.use_noise_aware:
            # Noise-aware subtraction scaled by guidance prior G:
            # In bright areas (G -> 0), subtraction is suppressed, preserving sharp details.
            # In dark areas (G -> 1), full noise residual is removed.
            R_clean = R_low - G * noise_residual
        else:
            # Ablation C: Generic global residual without adaptive noise awareness
            R_clean = R_low - noise_residual
            
        return torch.clamp(R_clean, 0.0, 1.0)


class ChromaticityRelightNet(nn.Module):
    """
    Predicts relit illumination with chromaticity preservation.
    Outputs:
        I_delta: 3-channel illumination map (B, 3, H, W) to address channel-wise color imbalance,
        avoiding uniform scalar brightness multiplication that causes color casts.
    """
    def __init__(self, channel=64, kernel_size=3):
        super(ChromaticityRelightNet, self).__init__()
        padding = kernel_size // 2

        # 4 inputs: R_clean (3) + L_low (1)
        self.conv0 = nn.Conv2d(4, channel, kernel_size, padding=padding)

        self.conv1 = nn.Conv2d(channel, channel, kernel_size, stride=2, padding=padding)
        self.conv2 = nn.Conv2d(channel, channel, kernel_size, stride=2, padding=padding)
        self.conv3 = nn.Conv2d(channel, channel, kernel_size, stride=2, padding=padding)

        self.relu = nn.ReLU(inplace=True)

        self.deconv1 = nn.Conv2d(channel, channel, kernel_size, padding=padding)
        self.deconv2 = nn.Conv2d(channel, channel, kernel_size, padding=padding)
        self.deconv3 = nn.Conv2d(channel, channel, kernel_size, padding=padding)

        self.feature_fusion = nn.Conv2d(channel * 3, channel, 1)
        
        # 3-channel illumination output (RGB illumination with shared baseline + color balance)
        self.output = nn.Conv2d(channel, 3, kernel_size=3, padding=1)

    def forward(self, input_L, input_R):
        input_im = torch.cat((input_R, input_L), dim=1)

        conv0 = self.conv0(input_im)
        conv1 = self.relu(self.conv1(conv0))
        conv2 = self.relu(self.conv2(conv1))
        conv3 = self.relu(self.conv3(conv2))

        up1 = F.interpolate(conv3, size=conv2.shape[2:], mode="nearest")
        deconv1 = self.relu(self.deconv1(up1)) + conv2

        up2 = F.interpolate(deconv1, size=conv1.shape[2:], mode="nearest")
        deconv2 = self.relu(self.deconv2(up2)) + conv1

        up3 = F.interpolate(deconv2, size=conv0.shape[2:], mode="nearest")
        deconv3 = self.relu(self.deconv3(up3)) + conv0

        deconv1_resize = F.interpolate(deconv1, size=deconv3.shape[2:], mode="nearest")
        deconv2_resize = F.interpolate(deconv2, size=deconv3.shape[2:], mode="nearest")

        feature_gather = torch.cat((deconv1_resize, deconv2_resize, deconv3), dim=1)
        feature_fusion = self.feature_fusion(feature_gather)
        output = self.output(feature_fusion)

        return torch.sigmoid(output)
