# Fast Local Image Editing with Distilled Diffusion Models

**100x faster than DDIM inversion. Runs on consumer GPUs.**

![Hero Image](figures/comparison_all_000000000069.png)

---

Traditional diffusion-based image editing methods like DDIM inversion and Prompt-to-Prompt are notoriously slow. On an RTX 3060 laptop GPU, these approaches can take **~10 minutes per image**. Even on datacenter GPUs like the A100, traditional inversion methods remain prohibitively slow for most applications, requiring 50-100 reverse diffusion steps plus expensive null-text optimization.

This project demonstrates that **distilled diffusion models** (SSD-1B) combined with **consistency models** (LCM) can achieve **~6 seconds per image** on the same consumer hardware—a **100x speedup**—while maintaining strong semantic accuracy. The key insight: without massive compute resources to train new models, we can still achieve dramatic performance gains through careful model selection, strategic integration, and precision optimization. For implementation details, see [IMPLEMENTATION.md](IMPLEMENTATION.md).

---

## Contents

- [Results at a Glance](#results-at-a-glance)
- [Architecture](#architecture)
- [Critical Optimizations](#critical-optimizations)
  - [FP16 Precision is Literally Free](#1-fp16-precision-is-literally-free)
  - [CPU Offloading: Intelligent, Not Slow](#2-cpu-offloading-intelligent-not-slow)
  - [Small ControlNet for SSD-1B](#3-small-controlnet-for-ssd-1b)
  - [Four-Step LCM Inference](#4-four-step-lcm-inference)
  - [Quality Fixes That Mattered](#5-quality-fixes-that-mattered)
- [Benchmark Methodology](#benchmark-methodology)
- [Consumer Hardware Performance](#consumer-hardware-performance)
- [Limitations](#limitations)
- [Future Directions](#future-directions)
- [System Configuration](#system-configuration)

---

## Results at a Glance

**Speed (RTX 3060 6GB Laptop):**

| Method | Time/Image | Speedup |
|--------|------------|---------|
| DDIM Prompt-to-Prompt | ~600s (10 min) | 1× |
| **Our Pipeline (SSD-1B FP16)** | **~6s** | **100×** |

**Quality (700 PIE-Bench images on A100):**

| Configuration | SSIM ↑ | LPIPS ↓ | CLIP ↑ | PSNR ↑ |
|--------------|--------|---------|--------|--------|
| DDIM P2P (Baseline) | 0.711 | 0.209 | 25.01 | 17.87 |
| **SSD-1B FP16 (Ours)** | 0.436 | 0.458 | **32.07** | 10.33 |
| SDXL FP16 | 0.464 | 0.406 | 31.90 | 11.29 |

The metrics reveal an interesting trade-off: our method sacrifices some structural similarity (SSIM) compared to the slow baseline, but achieves **28% higher CLIP score**, indicating better text-image alignment. For applications where speed matters and perfect pixel-level reconstruction isn't required, this is a favorable trade.

**Memory Efficiency:**

| Configuration | VRAM (A100) | Notes |
|--------------|-------------|-------|
| SDXL FP32 | 26.5 GB | Crashes on RTX 3060 |
| SDXL FP16 | 14.3 GB | 46% VRAM reduction |
| SSD-1B FP32 | 10.5 GB | Usable with offloading |
| **SSD-1B FP16** | **5.3 GB** | **49.5% VRAM reduction, fits on 6GB GPU** |

---

## Architecture

The pipeline combines three components: a **base diffusion model** (SDXL or its distilled variant SSD-1B), **Latent Consistency Models** for few-step inference, and **ControlNet** for structure preservation via Canny edge guidance.

```mermaid
graph TB
    subgraph Input
        A[Source Image] --> B[Canny Edge Detector]
        A --> C[VAE Encoder]
        D[Text Prompt] --> E[CLIP Text Encoder]
    end

    B --> F[Edge Map]
    C --> G[Latent Representation<br/>4×64×64]
    E --> H[Text Embeddings]

    subgraph "Guided Generation"
        F --> I[ControlNet<br/>Canny]
        G --> J[UNet + LCM LoRA<br/>4 steps, CFG=1.5]
        H --> J
        I -->|Structure Guidance| J
    end

    J --> K[Edited Latent]
    K --> L[VAE Decoder<br/>fp16-fix]
    L --> M[Edited Image]

    style J fill:#e1f5ff
    style I fill:#fff4e1
    style L fill:#ffe1f5
```

The choice of **SSD-1B** over SDXL was driven by memory constraints—at 50% smaller (1.8B vs 3.5B parameters), it fits comfortably on consumer GPUs. **LCM** (Latent Consistency Models) replaces the 50-step DDIM process with just 4 steps, providing a 12× inference speedup with minimal quality loss. **ControlNet** extracts Canny edges from the source image to guide generation, preventing catastrophic changes while allowing semantic edits.

Both SDXL and SSD-1B operate at **1024×1024 resolution**, a significant quality improvement over SD 1.5's 512×512. This higher resolution enables more detailed edits and better preservation of fine structures in the source image.

![SSD-1B vs SDXL](figures/comparison_sdxl_fp16_vs_ssd-1b_fp16_000000000061.png)
*SDXL FP16 preserves slightly finer details, but SSD-1B FP16 achieves comparable semantic accuracy 33% faster*

---

## Critical Optimizations

Getting this pipeline to run on a 6GB RTX 3060 required addressing multiple bottlenecks ([see full hardware specs](#system-configuration)). Here's what made the difference:

### 1. FP16 Precision is Literally Free

Using FP16 (`torch.float16`) cuts memory usage in half compared to FP32, with zero perceptual quality loss when paired with the `madebyollin/sdxl-vae-fp16-fix` VAE. This specialized VAE prevents the NaN issues and color artifacts that typically plague FP16 inference.

![SDXL FP16 vs FP32](figures/comparison_sdxl_fp16_vs_sdxl_fp32_000000000001.png)
*SDXL FP16 vs FP32: visually identical outputs, 46% VRAM savings*

The numbers back this up:
- **SSIM difference**: 0.00% (identical structure preservation)
- **LPIPS difference**: 0.25% (negligible perceptual change)
- **VRAM savings**: 49.5% for SSD-1B, 46.0% for SDXL

An early mistake was using `force_upcast=True` on the VAE, which defeated the purpose of the fp16-fix and caused green/purple color tints. Removing this flag and trusting the fp16-fix VAE resolved the issue entirely.

![SSD-1B FP16 vs FP32](figures/comparison_ssd-1b_fp16_vs_ssd-1b_fp32_000000000050.png)
*Even with the distilled model, FP16 maintains quality while offering significant speed/memory benefits*

### 2. CPU Offloading: Intelligent, Not Slow

Conventional wisdom suggests CPU offloading is "just slower." The reality is more nuanced. On GPUs with limited VRAM, **CPU offloading is intelligent memory management** that prevents a catastrophic failure mode: DRAM paging.

**RTX 3060 Performance (SSD-1B FP16):**
- With CPU offload enabled: **~6s per image**
- With CPU offload disabled: **~25s per image**

That's a **4.2× slowdown** when you'd expect offloading to hurt performance. What's happening? Without offloading, the model (5.3GB) fits barely within 6GB VRAM, but memory fragmentation and temporary allocations push it over the edge. The system falls back to DRAM paging, swapping model weights between GPU and system memory on every operation—a severe performance hit.

```mermaid
graph LR
    subgraph "With CPU Offload (6s)"
        A1[GPU 6GB] -->|Active UNet| A2[~4.5GB Used]
        A3[CPU RAM] -->|Text Encoder, VAE| A4[~5GB]
        A2 -.->|Swap Components| A4
    end

    subgraph "Without Offload (25s)"
        B1[GPU 6GB] -->|All Models| B2[~5.3GB + Overhead]
        B2 -->|Memory Overflow| B3[DRAM Paging]
        B3 -->|4.2x Slower| B4[Severe Slowdown]
    end

    style A2 fill:#90EE90
    style B3 fill:#FFB6C6
    style B4 fill:#FF6B6B
```

CPU offloading intelligently manages which components stay GPU-resident (the active UNet layers during inference) versus which can safely live in RAM (inactive text encoders, VAE). This keeps the working set small enough to avoid paging, resulting in 4× faster execution than the naive "no offload" approach.

When models exceed VRAM capacity entirely (e.g., SDXL FP32 on 6GB), offloading becomes essential—or the system crashes. On high-VRAM GPUs like the A100 (80GB), offloading is unnecessary and adds a 20-30% overhead, so we disable it for benchmark runs.

### 3. Small ControlNet for SSD-1B

Using the `controlnet-canny-sdxl-1.0-small` variant instead of the full ControlNet saves ~2GB VRAM with minimal quality impact. For SDXL benchmarks, we used the full ControlNet to maximize quality; for SSD-1B, the small variant was sufficient and enabled faster iteration during development.

### 4. Four-Step LCM Inference

LCM models are specifically optimized for 4 steps with low guidance scale (CFG=1.5). More steps provide diminishing returns and increase latency. This single change—DDIM's 50 steps to LCM's 4—accounts for most of the 12× speedup.

### 5. Quality Fixes That Mattered

- **Removed VAE tiling**: Caused color artifacts at tile boundaries; unnecessary with attention slicing
- **Disabled `force_upcast`**: Prevented the fp16-fix VAE from working correctly
- **Attention slicing**: Enabled only when CPU offloading is active (memory-constrained scenarios)

### 6. Resolution: 1024×1024 is Non-Negotiable

SDXL and SSD-1B are trained specifically for 1024×1024 resolution. An attempt to run at 512×512 for faster inference resulted in severely degraded "deepfried" outputs with poor quality. The models don't gracefully downscale—they require their native resolution to function properly. While this increases computation compared to SD 1.5 (512×512), the quality improvement and LCM speedup more than compensate for the extra pixels.

---

## Benchmark Methodology

All quality metrics come from **700 PIE-Bench v1 images** processed on a Google Colab A100 (80GB VRAM) with CPU offloading disabled. Four configurations were tested: SDXL and SSD-1B, each in FP32 and FP16, all using 4-step LCM inference with ControlNet guidance (CFG=1.5, ControlNet scale=0.5).

**Resolution Handling**: PIE-Bench images are 512×512, but SDXL/SSD-1B require 1024×1024 input. Images were upscaled to 1024×1024 for inference, then downscaled back to 512×512 for metric calculation to remain consistent with PIE-Bench baseline measurements.

PIE-Bench is a text-guided image editing benchmark covering diverse edit types: object replacement, attribute changes, positional edits, etc. We report six metrics:

- **SSIM** (Structural Similarity): Structure preservation vs source (0-1, higher better)
- **LPIPS** (Learned Perceptual Image Patch Similarity): Perceptual distance (0-1+, lower better)
- **CLIP Score**: Text-image alignment (0-100, higher better)
- **PSNR**: Signal quality in dB (higher better)
- **MSE**: Mean squared error (0-1, lower better)
- **DINO**: Semantic similarity via self-supervised ViT (0-1, lower better)

The DDIM Prompt-to-Prompt baseline was measured on the same hardware with 50 inference steps, standard CFG=7.5, and null-text optimization. **Note**: The 600s baseline timing was measured with 10 DDIM steps on the RTX 3060 (50 steps would take significantly longer); this represents a practical lower bound for traditional inversion methods.

### Full Results (700 images × 4 configurations = 2,800 total edits)

| Configuration | SSIM ↑ | LPIPS ↓ | CLIP ↑ | PSNR ↑ | MSE ↓ | DINO ↓ |
|--------------|--------|---------|--------|--------|-------|--------|
| DDIM P2P (Baseline) | 0.711 | 0.209 | 25.01 | 17.87 | 0.022 | 0.069 |
| SDXL FP32 | 0.464 | 0.405 | 31.90 | 11.29 | 0.085 | 0.043 |
| SDXL FP16 | 0.464 | 0.406 | 31.90 | 11.29 | 0.085 | 0.043 |
| SSD-1B FP32 | 0.436 | 0.458 | 32.10 | 10.35 | 0.111 | 0.052 |
| **SSD-1B FP16** | **0.436** | **0.458** | **32.07** | **10.33** | **0.111** | **0.052** |

Speed context: DDIM baseline takes ~600s/image on RTX 3060; our methods run 6-118s depending on config (5-100× faster). A100 benchmarks focus on quality, not speed, since ample VRAM eliminates offloading overhead.

![All Methods](figures/comparison_all_000000000037.png)
*All four configurations successfully edit while preserving composition*

FP16 produces **identical quality** to FP32 for both models. SDXL edges out SSD-1B on structure (SSIM: 0.464 vs 0.436, +6.4%) and perceptual quality (LPIPS: 0.405 vs 0.458, -11.6%), but SSD-1B wins slightly on CLIP score (32.07 vs 31.90, +0.5%) and is 33% faster on RTX 3060.

Interestingly, all fast methods achieve **higher CLIP scores** than the DDIM baseline (32.07 vs 25.01, +28%), despite lower SSIM. This reflects a fundamental difference: the baseline prioritizes pixel-level reconstruction, while our methods optimize for text alignment. DINO distance is also 25-38% lower for our methods, indicating better semantic preservation.

![SDXL Comparison](figures/comparison_sdxl_fp16_vs_ssd-1b_fp16_000000000100.png)
*Complex scene: both models handle semantic changes reliably*

The visual differences between FP16 and FP32 are imperceptible, validating the use of fp16-fix VAE. SDXL produces marginally sharper textures, but for interactive applications, SSD-1B's speed advantage outweighs the quality delta.

---

## Consumer Hardware Performance

All timing measurements below are from an **RTX 3060 Laptop GPU (6GB VRAM)** with CPU offloading enabled where necessary ([see full specs](#system-configuration)):

| Configuration | Time/Image | CPU Offload | Notes |
|--------------|------------|-------------|-------|
| **SSD-1B FP16** | **~6s** | Enabled | **Recommended: 100× faster than DDIM** |
| SSD-1B FP16 (no offload) | ~25s | Disabled | DRAM paging catastrophe |
| SSD-1B FP32 | ~118s | Enabled | Slower but usable |
| SDXL FP16 | ~113s | Enabled | Comparable to SSD-1B FP32 |
| SDXL FP32 | CRASH | N/A | Exceeds 6GB even with offload |

Only SSD-1B FP16 achieves real-time performance. The "no offload" configuration is particularly instructive: even though the model theoretically fits in 6GB, fragmentation causes DRAM paging, resulting in 4.2× slowdown. This underscores the importance of **GPU residency**—keeping the working set firmly within VRAM is more important than minimizing total model size.

---

## Limitations

While the pipeline achieves strong performance on most edits, certain failure modes persist:

### Color Handling Issues

The 4-step LCM inference sometimes struggles with precise color reproduction, particularly for unusual color combinations or specific color requests. This manifests as unexpected color shifts (pink/green tints) or incorrect color application:

![Color Issues 1](figures/comparison_all_000000000101.png)
*Example 1: Unintended color shifts in generated outputs*

![Color Issues 2](figures/comparison_all_000000000116.png)
*Example 2: Difficulty maintaining accurate color reproduction*

![Color Issues 3](figures/comparison_all_111000000001.png)
*Example 3: Pink and green color artifacts*

### Object Consistency and Distortions

The combination of ControlNet edge guidance and few-step diffusion can lead to object morphing or inconsistent details, especially in complex scenes:

![Distortion 1](figures/comparison_all_221000000001.png)
*Example 1: Object consistency issues with structural elements*

![Distortion 2](figures/comparison_all_322000000003.png)
*Example 2: Morphing and distortion in complex scenes*

![Distortion 3](figures/comparison_all_422000000003.png)
*Example 3: Structural distortions when editing detailed objects*

### Other Known Limitations

**Structure preservation trade-off**: ControlNet guides generation via edges, which preserves layout well but may limit drastic semantic changes (e.g., replacing a dog with a car).

**Quality ceiling**: Four-step LCM inference is optimized for speed, not maximum fidelity. Complex scenes with fine details may benefit from more steps, at the cost of proportionally slower inference.

**Resolution requirement**: SDXL and SSD-1B require 1024×1024 input resolution and don't support lower resolutions without severe quality degradation. This increases computational cost compared to SD 1.5 (512×512) but is necessary for the models to function properly.

**Edge detection sensitivity**: Canny edge extraction struggles with very cluttered or low-contrast images, producing poor guidance maps.

**Text rendering**: Like most diffusion models, the pipeline struggles with generating or editing text within images.

---

## Future Directions

- Benchmark on additional consumer GPUs (RTX 4060, 4070, etc.)
- Per-editing-type performance breakdown (object replacement vs attribute change)
- Comparison with other fast editing methods (InstructPix2Pix, MagicBrush)
- Img2Img pipeline variant (simpler, potentially less VRAM)
- SD 1.5 support (faster on limited hardware)
- Improved color consistency through fine-tuning or alternative schedulers

---

## System Configuration

**RTX 3060 Hardware (Performance Testing):**
- OS: Windows 11, WSL 2.6.1.0
- CPU: AMD Ryzen 7 5800H with Radeon Graphics (8C/16T, 3.2GHz base)
- RAM: 32GB DDR4
- GPU: NVIDIA GeForce RTX 3060 Laptop GPU (6GB GDDR6)
- PyTorch: 2.8.0+cu129
- Diffusers: 0.35.2
- Transformers: 4.57.1

**A100 Hardware (Quality Benchmarks):**
- GPU: NVIDIA A100 (80GB VRAM)
- Platform: Google Colab Pro+
- Purpose: 700 images × 4 configurations

---

## Citations

- **SDXL**: Podell, D., et al. (2023). "SDXL: Improving Latent Diffusion Models for High-Resolution Image Synthesis." [arXiv:2307.01952](https://arxiv.org/abs/2307.01952)
- **Latent Consistency Models**: Luo, S., et al. (2023). "Latent Consistency Models: Synthesizing High-Resolution Images with Few-Step Inference." [arXiv:2310.04378](https://arxiv.org/abs/2310.04378)
- **ControlNet**: Zhang, L., Rao, A., & Agrawala, M. (2023). "Adding Conditional Control to Text-to-Image Diffusion Models." [arXiv:2302.05543](https://arxiv.org/abs/2302.05543)
- **SSD-1B**: Segmind. "SSD-1B: A distilled 50% smaller SDXL model." [HuggingFace](https://huggingface.co/segmind/SSD-1B)
- **PIE-Bench**: Ju, C., et al. (2023). "Direct Inversion: Boosting Diffusion-based Editing with 3 Lines of Code." [arXiv:2310.01506](https://arxiv.org/abs/2310.01506)
