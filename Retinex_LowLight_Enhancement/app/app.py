"""
Interactive Streamlit Demonstration Application for Final-Year B.Tech CSE Project.
Title: Deep Learning-Based Low-Light Image Enhancement Using Retinex Networks
       with Illumination-Guided Noise-Aware Reflectance Refinement (IGNRR)

Features:
- Live inference using trained PyTorch checkpoints
- Side-by-side comparison (Low-Light Input | Baseline Retinex | Proposed IGNRR | Ground Truth)
- Component visualization: Reflectance R, Illumination L, Guidance Map G
- Real-time quantitative metric evaluation (PSNR, SSIM, Delta E 2000, RGB MAE)
- Direct download of enhanced images
- Theoretical explanation & architectural diagrams
"""

import os
import sys
import time
from io import BytesIO
import numpy as np
from PIL import Image
import streamlit as st
import torch

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from configs.config import (
    DATASET_DIR,
    CORRECTED_BASELINE_DIR,
    IMPROVED_CHECKPOINT_DIR,
    RESULTS_DIR,
    DEVICE
)
from utils.checkpoint import load_checkpoint
from utils.metrics import compute_psnr, compute_ssim, compute_color_metrics
from models.decom_net import DecomNet
from models.relight_net import RelightNet
from models.ignrr_net import build_ignrr_full


# Page Configuration
st.set_page_config(
    page_title="IGNRR Low-Light Image Enhancement",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F3F4F6;
        border-radius: 8px;
        padding: 12px;
        text-align: center;
        border-left: 4px solid #3B82F6;
    }
    .stDownloadButton button {
        width: 100%;
        background-color: #2563EB;
        color: white;
    }
</style>
""", unsafe_allow_html=True)


def safe_st_image(img, **kwargs):
    """Display an image safely across both old and new Streamlit versions."""
    try:
        st.image(img, use_container_width=True, **kwargs)
    except TypeError:
        st.image(img, use_column_width=True, **kwargs)


@st.cache_resource
def load_models():
    """
    Cached model loader to avoid re-instantiating networks on every rerun.
    """
    device = torch.device("cpu")  # CPU for reliable local web demo
    
    # 1. Baseline Models
    decom = DecomNet().to(device)
    relight = RelightNet().to(device)
    decom_path = os.path.join(CORRECTED_BASELINE_DIR, "decom_best.pth")
    relight_path = os.path.join(CORRECTED_BASELINE_DIR, "relight_best.pth")
    
    if os.path.exists(decom_path):
        load_checkpoint(decom, decom_path, device=device)
    if os.path.exists(relight_path):
        load_checkpoint(relight, relight_path, device=device)
    decom.eval()
    relight.eval()

    # 2. Proposed IGNRR Model
    ignrr_model = build_ignrr_full().to(device)
    ignrr_path = os.path.join(IMPROVED_CHECKPOINT_DIR, "ignrr_best.pth")
    if not os.path.exists(ignrr_path):
        ignrr_path = os.path.join(IMPROVED_CHECKPOINT_DIR, "ignrr_latest.pth")
        
    ignrr_loaded = False
    if os.path.exists(ignrr_path):
        try:
            load_checkpoint(ignrr_model, ignrr_path, device=device)
            ignrr_loaded = True
        except Exception as e:
            st.sidebar.warning(f"Error loading IGNRR checkpoint: {e}")
            
    ignrr_model.eval()
    return decom, relight, ignrr_model, ignrr_loaded


def run_baseline_inference(low_arr, decom, relight):
    t0 = time.time()
    low_t = torch.from_numpy(low_arr).permute(2, 0, 1).unsqueeze(0)
    with torch.no_grad():
        R_low, L_low = decom(low_t)
        I_delta = relight(L_low, R_low)
        enhanced = torch.clamp(R_low * I_delta.repeat(1, 3, 1, 1), 0.0, 1.0)
    latency = (time.time() - t0) * 1000.0
    return {
        "enhanced": enhanced[0].permute(1, 2, 0).numpy(),
        "R": R_low[0].permute(1, 2, 0).numpy(),
        "L": L_low[0, 0].numpy(),
        "I_delta": I_delta[0, 0].numpy(),
        "latency_ms": latency
    }


def run_ignrr_inference(low_arr, ignrr_model):
    t0 = time.time()
    low_t = torch.from_numpy(low_arr).permute(2, 0, 1).unsqueeze(0)
    with torch.no_grad():
        outs = ignrr_model(low_t)
    latency = (time.time() - t0) * 1000.0
    return {
        "enhanced": outs["enhanced"][0].permute(1, 2, 0).numpy(),
        "R_low": outs["R_low"][0].permute(1, 2, 0).numpy(),
        "R_refined": outs["R_refined"][0].permute(1, 2, 0).numpy(),
        "L_low": outs["L_low"][0, 0].numpy(),
        "guidance": (1.0 - outs["L_low"][0, 0]).numpy(),
        "I_delta": outs["I_delta"][0].permute(1, 2, 0).numpy(),
        "latency_ms": latency
    }


def main():
    # Header
    st.markdown('<div class="main-header">✨ Low-Light Image Enhancement with IGNRR</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">'
        'Retinex Deep Learning Network with <b>Illumination-Guided Noise-Aware Reflectance Refinement</b> | '
        'B.Tech CSE Capstone Project'
        '</div>',
        unsafe_allow_html=True
    )

    decom, relight, ignrr_model, ignrr_loaded = load_models()

    # Sidebar Controls
    st.sidebar.header("🕹️ Controls & Settings")
    input_source = st.sidebar.radio(
        "Select Image Input Mode:",
        ["LOL-v1 Benchmark Test Pair", "Upload Custom Low-Light Image"]
    )

    low_image = None
    high_image = None
    image_name = "sample"

    test_low_dir = os.path.join(DATASET_DIR, "test", "low")
    test_high_dir = os.path.join(DATASET_DIR, "test", "high")

    if input_source == "LOL-v1 Benchmark Test Pair":
        if os.path.exists(test_low_dir):
            test_files = sorted(os.listdir(test_low_dir))
            selected_file = st.sidebar.selectbox("Choose Benchmark Test Image:", test_files, index=0)
            image_name = selected_file
            low_path = os.path.join(test_low_dir, selected_file)
            high_path = os.path.join(test_high_dir, selected_file)
            low_image = Image.open(low_path).convert("RGB")
            if os.path.exists(high_path):
                high_image = Image.open(high_path).convert("RGB")
        else:
            st.sidebar.error("Test dataset folder not found.")
    else:
        uploaded_file = st.sidebar.file_uploader("Upload Low-Light Image (PNG/JPG):", type=["png", "jpg", "jpeg"])
        if uploaded_file is not None:
            low_image = Image.open(uploaded_file).convert("RGB")
            image_name = uploaded_file.name

        uploaded_gt = st.sidebar.file_uploader("Optional: Upload Ground Truth (Normal Light):", type=["png", "jpg", "jpeg"])
        if uploaded_gt is not None:
            high_image = Image.open(uploaded_gt).convert("RGB")

    show_intermediates = st.sidebar.checkbox("Show Decomposition & Guidance Maps", value=True)

    if low_image is None:
        st.info("👈 Please select or upload a low-light image from the sidebar to begin enhancement.")
        return

    low_arr = np.asarray(low_image, dtype=np.float32) / 255.0
    high_arr = np.asarray(high_image, dtype=np.float32) / 255.0 if high_image is not None else None

    # Perform Inferences
    base_out = run_baseline_inference(low_arr, decom, relight)
    ignrr_out = run_ignrr_inference(low_arr, ignrr_model) if ignrr_loaded else None

    # Top Comparison Section
    st.subheader("🖼️ Comparative Enhancement Results")

    cols = st.columns(4 if high_arr is not None else 3)
    with cols[0]:
        st.markdown("**1. Low-Light Input**")
        safe_st_image(low_image)
        st.caption(f"Dimensions: {low_image.width} × {low_image.height}")

    with cols[1]:
        st.markdown(f"**2. Baseline RetinexNet** ({base_out['latency_ms']:.1f} ms)")
        base_pil = Image.fromarray((base_out["enhanced"] * 255).astype(np.uint8))
        safe_st_image(base_pil)
        base_buf = BytesIO()
        base_pil.save(base_buf, format="PNG")
        st.download_button(
            "⬇️ Download Baseline Output",
            data=base_buf.getvalue(),
            file_name=f"baseline_{image_name}",
            mime="image/png",
            key="dl_base"
        )

    with cols[2]:
        if ignrr_out is not None:
            st.markdown(f"**3. Proposed IGNRR (Ours)** ({ignrr_out['latency_ms']:.1f} ms)")
            ignrr_pil = Image.fromarray((ignrr_out["enhanced"] * 255).astype(np.uint8))
            safe_st_image(ignrr_pil)
            ignrr_buf = BytesIO()
            ignrr_pil.save(ignrr_buf, format="PNG")
            st.download_button(
                "⬇️ Download IGNRR Output",
                data=ignrr_buf.getvalue(),
                file_name=f"ignrr_enhanced_{image_name}",
                mime="image/png",
                key="dl_ignrr"
            )
        else:
            st.markdown("**3. Proposed IGNRR**")
            st.warning("Model checkpoint not yet available. Train model to activate.")

    if high_arr is not None:
        with cols[3]:
            st.markdown("**4. Ground Truth (Reference)**")
            safe_st_image(high_image)
            st.caption("Normal-light benchmark")

    # Quantitative Metrics Scorecard (if Ground Truth exists)
    if high_arr is not None:
        st.divider()
        st.subheader("📊 Quantitative Image Quality Metrics")
        
        # Resize high_arr if dimension slightly differs
        if high_arr.shape != low_arr.shape:
            high_image = high_image.resize((low_image.width, low_image.height))
            high_arr = np.asarray(high_image, dtype=np.float32) / 255.0

        p_base = compute_psnr(base_out["enhanced"], high_arr)
        s_base = compute_ssim(base_out["enhanced"], high_arr)
        c_base = compute_color_metrics(base_out["enhanced"], high_arr)

        if ignrr_out is not None:
            p_ignrr = compute_psnr(ignrr_out["enhanced"], high_arr)
            s_ignrr = compute_ssim(ignrr_out["enhanced"], high_arr)
            c_ignrr = compute_color_metrics(ignrr_out["enhanced"], high_arr)

            m_cols = st.columns(4)
            with m_cols[0]:
                st.metric(
                    label="PSNR (Higher is better)",
                    value=f"{p_ignrr:.2f} dB",
                    delta=f"{p_ignrr - p_base:+.2f} dB vs Baseline"
                )
            with m_cols[1]:
                st.metric(
                    label="SSIM (Higher is better)",
                    value=f"{s_ignrr:.4f}",
                    delta=f"{s_ignrr - s_base:+.4f} vs Baseline"
                )
            with m_cols[2]:
                st.metric(
                    label="Delta E 2000 (Lower is better)",
                    value=f"{c_ignrr['delta_e_2000']:.2f}",
                    delta=f"{c_ignrr['delta_e_2000'] - c_base['delta_e_2000']:-.2f} (Color Drift)",
                    delta_color="inverse"
                )
            with m_cols[3]:
                st.metric(
                    label="RGB MAE (Lower is better)",
                    value=f"{c_ignrr['rgb_mae']:.4f}",
                    delta=f"{c_ignrr['rgb_mae'] - c_base['rgb_mae']:-.4f}",
                    delta_color="inverse"
                )
        else:
            m_cols = st.columns(4)
            m_cols[0].metric("Baseline PSNR", f"{p_base:.2f} dB")
            m_cols[1].metric("Baseline SSIM", f"{s_base:.4f}")
            m_cols[2].metric("Baseline Delta E 2000", f"{c_base['delta_e_2000']:.2f}")
            m_cols[3].metric("Baseline RGB MAE", f"{c_base['rgb_mae']:.4f}")

    # Decomposition & Illumination Guidance Maps
    if show_intermediates and ignrr_out is not None:
        st.divider()
        st.subheader("🔬 Retinex Decomposition & Illumination Guidance Pipeline")
        p_cols = st.columns(4)

        with p_cols[0]:
            st.markdown("**Estimated Illumination ($L_{low}$)**")
            safe_st_image(ignrr_out["L_low"], clamp=True)
            st.caption("Low dynamic range light component")

        with p_cols[1]:
            st.markdown("**Illumination Guidance Map ($G = 1 - L$)**")
            safe_st_image(ignrr_out["guidance"], clamp=True)
            st.caption("Spatial noise-risk prior (white = heavy noise)")

        with p_cols[2]:
            st.markdown("**Raw DecomNet Reflectance ($R_{low}$)**")
            safe_st_image(ignrr_out["R_low"], clamp=True)
            st.caption("Contains amplified dark sensor noise")

        with p_cols[3]:
            st.markdown("**Refined Reflectance ($R_{refined}$)**")
            safe_st_image(ignrr_out["R_refined"], clamp=True)
            st.caption("Noise suppressed, edges preserved")

    # Technical Explanation & Viva Reference
    st.divider()
    with st.expander("📚 Research Motivation & Mathematical Formulation (For Project Evaluation / Viva)"):
        st.markdown(r"""
        ### 1. Retinex Theory Foundation
        According to Retinex theory (Land & McCann), an observed image $I$ is the pixel-wise product of surface reflectance $R$ and environmental illumination $L$:
        $$I(x, y) = R(x, y) \cdot L(x, y)$$
        - **Reflectance ($R$):** Invariant intrinsic property representing true object color and fine structural textures.
        - **Illumination ($L$):** Represents non-uniform light distribution across the scene.

        ### 2. The Baseline Flaw: Noise Amplification
        In real CMOS/CCD sensors under extreme low light, sensor noise $n$ dominates dark pixels:
        $$I_{low} = R \cdot L_{low} + n$$
        When DecomNet isolates $R_{low}$, the noise is amplified by the inverse illumination:
        $$R_{low} \approx R + \frac{n}{L_{low}}$$
        Standard Retinex models (Wei et al. 2018) multiply $R_{low}$ directly by boosted illumination $I_\delta$:
        $$I_{enh} = R_{low} \cdot I_\delta = \left(R + \frac{n}{L_{low}}\right) \cdot I_\delta = R \cdot I_\delta + \mathbf{\left(\frac{I_\delta}{L_{low}}\right) n}$$
        Since $\frac{I_\delta}{L_{low}} \approx 8\times \text{ to } 15\times$, noise in shadow areas explodes, creating severe grain and color distortion ($\Delta E_{2000} \approx 16.5$).

        ### 3. Proposed Contribution: Illumination-Guided Noise-Aware Reflectance Refinement (IGNRR)
        - **Spatial Guidance Prior:** The illumination map $L_{low}$ directly predicts where noise is concentrated. We define the guidance map $G(x, y) = 1.0 - L_{low}(x, y)$.
        - **Adaptive Residual Refinement:** The refinement network subtracts noise proportional to $G$:
          $$R_{refined} = \text{clamp}\left(R_{low} - G \odot \hat{N}_R, 0, 1\right)$$
          In bright areas ($G \to 0$), original texture details are untouched. In pitch-dark areas ($G \to 1$), noise is aggressively filtered.
        - **Cosine Color Consistency Loss:** Aligns RGB chromaticity vectors with ground-truth, cutting Delta E 2000 and preventing yellow/blue color cast.
        """)


if __name__ == "__main__":
    main()
