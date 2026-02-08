# 🔬 X-Ray Shadow & Illumination Artifact Removal using U-Net

A deep learning–based pipeline for removing shadows and illumination artifacts from industrial X-ray images of electrical components using a multi-phase trained U-Net architecture.

This project was developed as part of research work at **National Taiwan University (NTU)** for improving structural clarity in X-ray inspection and defect analysis of PCB components.

---

## 🧠 Problem Statement

Industrial X-ray images of components suffer from:

- Low-frequency illumination gradients
- Severe shadow artifacts
- Contrast inconsistencies across components
- Loss of structural clarity required for defect analysis

Traditional image processing (morphology, filtering, histogram methods) fails to generalize across component shapes (BGA, pads, traces, IC bodies, etc.).

This project solves the problem using a **learning-based approach** instead of handcrafted post-processing.

---

## 🏗️ Architecture Overview

Input X-ray → Tiled U-Net Inference → Shadow-Free Output


The U-Net is trained in **3 progressive phases** to learn:

1. General shadow removal
2. Illumination normalization
3. Adaptation to real TRI component images

No heavy post-processing is required at inference.

---

## 🧪 Training Strategy (3-Phase Learning)

| Phase | Dataset | Objective |
|------|---------|-----------|
| Phase 1 | Custom synthetic shadow dataset | Learn basic shadow removal |
| Phase 2 | ISTD dataset | Learn illumination normalization + shadow masks |
| Phase 3 | Real TRI X-ray patches | Adapt to real component geometry |

Loss used: **L1 + SSIM hybrid loss**

---


---

## 🚀 Inference Pipeline

The model uses tiled inference to handle high-resolution X-ray images.

```python
from inference_tiled import test_image_tiled

output = test_image_tiled(
    image_path="input.jpg",
    tile_size=512,
    overlap=64
)
