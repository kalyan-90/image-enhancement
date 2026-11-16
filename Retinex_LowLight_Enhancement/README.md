# Deep Learning-Based Low-Light Image Enhancement Using Retinex Networks with Illumination-Guided Noise-Aware Reflectance Refinement (IGNRR)

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)



---

## 📌 Abstract

Under-exposed and low-light images suffer from degraded visibility, severe sensor noise, color distortion, and non-uniform dynamic range. Traditional deep Retinex methods decompose an image into reflectance $R$ and illumination $L$ ($I = R \times L$). However, because dark regions satisfy $I_{low} = R \cdot L_{low} + n$, decomposing extreme shadows amplifies the sensor noise inversely ($R_{low} \approx R + \frac{n}{L_{low}}$). Subsequent illumination boosting magnifies this noise up to $15\times$, leading to unnatural grain and severe color distortion.

This project introduces **Illumination-Guided Noise-Aware Reflectance Refinement (IGNRR)**. By leveraging the estimated illumination map as a spatial noise-risk guidance prior ($G = 1 - L_{low}$), IGNRR adaptively applies residual noise filtering to dark shadow regions while preserving sharp edges and textures in well-lit regions. Combined with a multi-channel chromaticity-aware relighting network and a cosine color consistency loss, the proposed architecture significantly improves structural clarity, reduces $\Delta E_{2000}$ perceptual color drift, and suppresses shadow noise.

---

## 🏗️ System Architecture

```
                                      ┌────────────────────────────────────────────────────────┐
                                      │              IGNRR Refinement Sub-Network              │
                                      │                                                        │
Low-Light Input I_low ──► [ DecomNet ]─┬─► L_low ─► [ Guidance: G = 1-L ] ──┐                  │
                                      │                                     ▼                  │
                                      └─► R_low ──────────────────────► [ NARB Blocks ] ───────┤
                                                                            │                  │
                                                                            ▼                  │
                                                                 Clean Reflectance R_refined   │
                                                                            │                  │
                                                                            ▼                  │
                                           L_low + R_refined ─► [ Chromaticity RelightNet ]    │
                                                                            │                  │
                                                                            ▼                  │
                                                                Relit Illumination I_delta     │
                                                                            │                  │
                                                                            ▼                  │
                                                              Enhanced Output = R_refined * I_delta
                                      └────────────────────────────────────────────────────────┘
```

---

## 📂 Project Organization

```text
Retinex_LowLight_Enhancement/
├── dataset/
│   └── LOL/
│       ├── train/           # 485 training pairs (437 train, 48 val)
│       │   ├── high/
│       │   └── low/
│       └── test/            # 15 isolated benchmark testing pairs
│           ├── high/
│           └── low/
├── models/
│   ├── decom_net.py         # Baseline decomposition network
│   ├── relight_net.py       # Baseline relighting network
│   ├── retinex_net.py       # Baseline end-to-end wrapper
│   ├── modules.py           # Novel IGNRR modules (Guidance, NARB, Attention)
│   └── ignrr_net.py         # Complete proposed IGNRR architecture & ablations
├── utils/
│   ├── dataset.py           # LOL-v1 PyTorch dataset loader with clean partitioning
│   ├── metrics.py           # PSNR, SSIM, Delta E 2000, RGB MAE, LAB errors
│   ├── losses.py            # Baseline RetinexLoss & Multi-objective IGNRRLoss
│   └── checkpoint.py        # Safe, unified checkpoint loader & saver
├── configs/
│   ├── config.py            # Global paths, hyperparameters, and device configs
│   └── train_val_split.json # Exact deterministic train (437) / val (48) split
├── training/
│   └── train_ignrr.py       # Resumable training script with auto validation & logging
├── evaluation/
│   ├── evaluate.py          # Standalone benchmark evaluation on test dataset
│   └── visualize.py         # Comparative side-by-side plot generator
├── checkpoints/
│   ├── corrected_baseline/  # Verified baseline checkpoints (decom_best.pth, relight_best.pth)
│   └── improved/            # Proposed IGNRR checkpoints (ignrr_best.pth)
├── experiments/             # CSV experimental logs and validation metric records
├── results/
│   ├── baseline/            # 15 baseline enhanced benchmark outputs
│   └── comparisons/         # Multi-panel visual comparison figures
├── app/
│   └── app.py               # Interactive Streamlit demonstration application
├── docs/
│   ├── architecture.md      # Mathematical derivation and design details
│   └── viva_prep.md         # Comprehensive Viva Voce defense Q&A guide
├── requirements.txt         # Project dependencies
└── README.md                # Project documentation
```

---

## 📊 Experimental Results (LOL-v1 Benchmark)

Evaluated under identical conditions across the isolated 15 test pairs:

| Experiment | Model Architecture | PSNR (dB) ↑ | SSIM ↑ | $\Delta E_{2000}$ ↓ | RGB MAE ↓ | Latency (ms) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **Input** | Raw Low-Light Observation | 7.77 | 0.1914 | 28.45 | 0.3842 | — |
| **A** | Corrected Retinex Baseline | 17.08 | 0.6950 | 16.46 | 0.1297 | ~840 ms |
| **B** | **Proposed IGNRR (Full)** | **18.92** | **0.7840** | **9.85** | **0.0812** | ~920 ms |
| **C** | Ablation: w/o Noise-Awareness | 17.65 | 0.7210 | 14.12 | 0.1105 | ~890 ms |
| **D** | Ablation: w/o Illumination Guidance | 17.40 | 0.7085 | 15.30 | 0.1190 | ~880 ms |

---

## 🚀 Getting Started

### 1. Installation
Clone the repository and install required packages:
```bash
git clone https://github.com/pravallikamothukuri4/Retinex_LowLight_Enhancement.git
cd Retinex_LowLight_Enhancement
pip install -r requirements.txt
```

### 2. Run the Interactive Streamlit Demo
Launch the web interface for live side-by-side demonstration:
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

### 3. Evaluate Checkpoints on Test Benchmark
Run the standalone evaluation pipeline:
```bash
python evaluation/evaluate.py --device cpu
```

### 4. Train the IGNRR Model
```bash
# Local training (CPU / GPU auto-detected)
python training/train_ignrr.py --epochs 25 --batch_size 8 --device cuda

# Train ablation models
python training/train_ignrr.py --model_type no_noise_aware --epochs 25
python training/train_ignrr.py --model_type no_illum_guidance --epochs 25
```

