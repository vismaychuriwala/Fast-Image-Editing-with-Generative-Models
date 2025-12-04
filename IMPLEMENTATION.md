# Implementation Guide

This document provides technical details for running and reproducing the fast image editing pipeline.

For the research report and benchmark results, see [README.md](README.md).

---

## Table of Contents

- [Installation](#installation)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Usage Examples](#usage-examples)
- [Configuration Options](#configuration-options)
- [Generating Visualizations](#generating-visualizations)
- [Reproducing Benchmark Results](#reproducing-benchmark-results)

---

## Installation

### Requirements

- Python 3.8+
- CUDA-enabled GPU (6GB+ VRAM recommended)
- ~20GB disk space for models

### Setup

```bash
# Clone repository
git clone <repository-url>
cd project

# Install dependencies
pip install -r requirements.txt

# Verify installation (optional)
python test_setup.py
```

### GPU Requirements

| Configuration | Minimum VRAM (no offload) | With CPU Offload |
|--------------|---------------------------|------------------|
| SSD-1B FP16  | 4 GB                      | 4 GB (offload not needed) |
| SDXL FP16    | 8 GB                      | 6 GB (offload required) |
| SSD-1B FP32  | 10 GB                     | 6 GB (offload required) |
| SDXL FP32    | 16 GB                     | 8 GB (offload required) |

**For RTX 3060 (6GB)**: Only SSD-1B FP16 can run without CPU offloading. SDXL FP16 and SSD-1B FP32 require CPU offloading (~25% slower). SDXL FP32 does not fit.

---

## Project Structure

```
project/
├── src/
│   ├── __init__.py
│   ├── pipeline.py          # FastEditor class (main pipeline)
│   └── metrics.py           # Evaluation metrics (SSIM, LPIPS, CLIP, etc.)
│
├── data/
│   └── PIE-Bench_v1/        # Dataset (gitignored)
│       ├── annotation_images/
│       └── mapping_file.json
│
├── outputs/                 # Generated images (gitignored)
│   ├── single/              # Single image outputs
│   │   ├── edited/{model}_{precision}/
│   │   └── comparisons/{model}_{precision}/
│   └── batch/               # Batch processing outputs
│       ├── edited/{model}_{precision}/
│       └── comparisons/{model}_{precision}/
│
├── results/                 # Evaluation results
│   ├── sdxl_fp32/
│   │   ├── metrics.csv
│   │   └── summary.json
│   ├── sdxl_fp16/
│   ├── ssd-1b_fp32/
│   └── ssd-1b_fp16/
│
├── figures/                 # Visual comparisons for documentation
│   ├── comparison_sdxl_fp16_vs_sdxl_fp32_*.png
│   ├── comparison_ssd-1b_fp16_vs_ssd-1b_fp32_*.png
│   ├── comparison_sdxl_fp16_vs_ssd-1b_fp16_*.png
│   └── comparison_all_*.png
│
├── plotting/                # Visualization tools
│   └── compare_methods.py   # Flexible comparison script
│
├── run_single_image.py      # Single image editing
├── run_batch.py             # Batch processing
├── evaluate.py              # Evaluation pipeline
├── requirements.txt
├── README.md                # Research report
└── IMPLEMENTATION.md        # This file
```

---

## Quick Start

### Single Image Editing

Edit a single image with a text prompt:

```bash
python run_single_image.py \
    --image data/PIE-Bench_v1/annotation_images/0_random_140/000000000000.jpg \
    --prompt "a slanted rusty mountain bicycle on the road in front of a building" \
    --model ssd-1b \
    --compute_metrics \
    --show_plot
```

**Output:**
- Edited image saved to `outputs/single/edited/ssd-1b_fp16/`
- Comparison plot saved to `outputs/single/comparisons/ssd-1b_fp16/`

### Batch Processing

Process multiple images from PIE-Bench:

```bash
python run_batch.py \
    --model ssd-1b \
    --num_images 50 \
    --steps 4 \
    --guidance 1.5 \
    --control_scale 0.5 \
    --save_comparisons \
    --skip_existing
```

**Output:**
- Edited images saved to `outputs/batch/edited/ssd-1b_fp16/`
- Preserves PIE-Bench directory structure

### Evaluation

Compute metrics for batch outputs:

```bash
python evaluate.py \
    --outputs_dir outputs/batch/edited/ssd-1b_fp16 \
    --results_file results/ssd-1b_fp16/metrics.csv \
    --summary_file results/ssd-1b_fp16/summary.json
```

**Metrics computed:**
- SSIM (Structural Similarity)
- LPIPS (Perceptual Distance)
- CLIP Score (Text-Image Alignment)
- PSNR (Signal Quality)
- MSE (Pixel Difference)
- DINO Distance (Semantic Similarity)

---

## Usage Examples

### Example 1: Fast Editing on Consumer GPU (6GB)

```bash
# SSD-1B FP16 - only config that runs on RTX 3060 without CPU offloading
python run_single_image.py \
    --image path/to/image.jpg \
    --prompt "your editing prompt" \
    --model ssd-1b \
    --no_cpu_offload \
    --steps 4 \
    --guidance 1.5 \
    --control_scale 0.5 \
    --seed 42
```

**Performance:** ~4 seconds/image on RTX 3060, ~3.8GB VRAM

**Note:** This is the ONLY configuration that fits on RTX 3060 6GB without CPU offloading.

### Example 2: Maximum Quality on High-End GPU

```bash
# SDXL FP32 - best quality (requires 16GB+ VRAM or A100)
python run_single_image.py \
    --image path/to/image.jpg \
    --prompt "your editing prompt" \
    --model sdxl \
    --quality_mode \
    --no_cpu_offload \
    --steps 4 \
    --guidance 1.5 \
    --control_scale 0.5 \
    --seed 42
```

**Performance:** ~4.5 seconds/image on A100, ~11.2GB VRAM

**Note:** Does NOT fit on RTX 3060 6GB even with CPU offloading. Requires 16GB+ VRAM or datacenter GPU.

### Example 3: Batch Processing with Filtering

```bash
# Process specific editing types
python run_batch.py \
    --model sdxl \
    --editing_types 0 1 2 \
    --num_images 100 \
    --save_comparisons

# Process specific image IDs
python run_batch.py \
    --model ssd-1b \
    --image_ids img001 img002 img003

# Resume interrupted batch
python run_batch.py \
    --model sdxl \
    --num_images 1000 \
    --skip_existing
```

---

## Configuration Options

### Model Selection (`--model`)

| Model   | Size    | Speed  | Quality | VRAM (FP16) | Best For |
|---------|---------|--------|---------|-------------|----------|
| ssd-1b  | Smaller | Faster | Good    | ~4 GB       | Real-time, consumer GPUs |
| sdxl    | Larger  | Slower | Better  | ~6 GB       | Quality-critical applications |

### Precision (`--full_precision` / `--quality_mode`)

| Flag | Precision | VRAM Usage | Speed | Quality |
|------|-----------|------------|-------|---------|
| (default) | FP16 | 1x | Faster | Excellent (with fp16-fix VAE) |
| `--full_precision` | FP32 | ~2x | ~25% slower | Marginally better (<1% difference) |
| `--quality_mode` | FP32 + Full ControlNet | ~2x | ~30% slower | Maximum quality |

**Recommendation:** Always use FP16 on consumer GPUs - quality difference is negligible.

### ControlNet Scale (`--control_scale`)

Controls structure preservation strength:

| Value | Effect |
|-------|--------|
| 0.3   | More editing freedom, less structure preservation |
| 0.5   | Balanced (default, recommended) |
| 0.7   | Strong structure preservation, limited editing |

### Guidance Scale (`--guidance`)

Controls prompt adherence:

| Value | Effect |
|-------|--------|
| 1.0   | More diverse, creative results |
| 1.5   | Balanced (default for LCM) |
| 2.0   | Stronger prompt following (may reduce quality) |

### Inference Steps (`--steps`)

| Steps | Speed | Quality |
|-------|-------|---------|
| 4     | Fastest | Good (LCM optimized) |
| 8     | 2x slower | Slightly better |
| 16+   | Not recommended | Diminishing returns for LCM |

**Recommendation:** Use 4 steps - LCM models are optimized for few-step inference.

### Canny Edge Thresholds

```bash
--canny_low 100    # Low threshold (default)
--canny_high 200   # High threshold (default)
```

Lower values detect more edges, higher values detect only strong edges.

### Memory Optimization

```bash
# Enable CPU offloading (for limited VRAM)
# Default: enabled
--no_cpu_offload   # Disable for faster inference (needs more VRAM)

# Use small ControlNet variant
# Default for SSD-1B, use --full_controlnet for full variant
--full_controlnet  # Use full-size ControlNet (needs ~2GB more VRAM)
```

---

## Generating Visualizations

### Compare All Methods

```bash
python plotting/compare_methods.py 000000000000
```

Output: `figures/comparison_all_000000000000.png`

### Compare Precision (FP16 vs FP32)

```bash
# SDXL precision comparison
python plotting/compare_methods.py 000000000000 --methods sdxl_fp16 sdxl_fp32

# SSD-1B precision comparison
python plotting/compare_methods.py 000000000005 --methods ssd-1b_fp16 ssd-1b_fp32
```

### Compare Models (SDXL vs SSD-1B)

```bash
python plotting/compare_methods.py 000000000010 --methods sdxl_fp16 ssd-1b_fp16
```

### Custom Output Directory

```bash
python plotting/compare_methods.py 000000000000 --output_dir my_figures
```

---

## Reproducing Benchmark Results

To reproduce the 700-image benchmarks reported in [README.md](README.md):

### Step 1: Process Images with All Configurations

```bash
# Configuration 1: SDXL FP32
python run_batch.py \
    --model sdxl \
    --quality_mode \
    --num_images 700 \
    --steps 4 \
    --guidance 1.5 \
    --control_scale 0.5 \
    --no_cpu_offload \
    --save_comparisons \
    --skip_existing

# Configuration 2: SDXL FP16
python run_batch.py \
    --model sdxl \
    --num_images 700 \
    --steps 4 \
    --guidance 1.5 \
    --control_scale 0.5 \
    --no_cpu_offload \
    --save_comparisons \
    --skip_existing

# Configuration 3: SSD-1B FP32
python run_batch.py \
    --model ssd-1b \
    --full_precision \
    --num_images 700 \
    --steps 4 \
    --guidance 1.5 \
    --control_scale 0.5 \
    --no_cpu_offload \
    --save_comparisons \
    --skip_existing

# Configuration 4: SSD-1B FP16 (fastest)
python run_batch.py \
    --model ssd-1b \
    --num_images 700 \
    --steps 4 \
    --guidance 1.5 \
    --control_scale 0.5 \
    --no_cpu_offload \
    --save_comparisons \
    --skip_existing
```

### Step 2: Evaluate Each Configuration

```bash
# Evaluate SDXL FP32
python evaluate.py \
    --outputs_dir outputs/batch/edited/sdxl_fp32 \
    --results_file results/sdxl_fp32/metrics.csv \
    --summary_file results/sdxl_fp32/summary.json

# Evaluate SDXL FP16
python evaluate.py \
    --outputs_dir outputs/batch/edited/sdxl_fp16 \
    --results_file results/sdxl_fp16/metrics.csv \
    --summary_file results/sdxl_fp16/summary.json

# Evaluate SSD-1B FP32
python evaluate.py \
    --outputs_dir outputs/batch/edited/ssd-1b_fp32 \
    --results_file results/ssd-1b_fp32/metrics.csv \
    --summary_file results/ssd-1b_fp32/summary.json

# Evaluate SSD-1B FP16
python evaluate.py \
    --outputs_dir outputs/batch/edited/ssd-1b_fp16 \
    --results_file results/ssd-1b_fp16/metrics.csv \
    --summary_file results/ssd-1b_fp16/summary.json
```

### Step 3: View Results

Results are saved to:
- `results/{model}_{precision}/metrics.csv` - Per-image metrics
- `results/{model}_{precision}/summary.json` - Aggregate statistics

---

## Troubleshooting

### Out of Memory (OOM)

1. **Use smaller model:** Switch from SDXL → SSD-1B
2. **Enable CPU offloading:** Remove `--no_cpu_offload` flag
3. **Use FP16:** Remove `--full_precision` flag
4. **Use small ControlNet:** Remove `--full_controlnet` flag
5. **Close other GPU processes**

### Poor Editing Quality

1. **Adjust ControlNet scale:** Try `--control_scale 0.3` to `0.7`
2. **Rephrase prompt:** Be more descriptive
3. **Adjust guidance:** Try `--guidance 1.0` to `2.0`
4. **Check Canny edges:** Adjust `--canny_low` and `--canny_high`

### Slow Performance

1. **Disable CPU offloading:** Use `--no_cpu_offload` (if VRAM allows)
2. **Use SSD-1B:** Faster than SDXL
3. **Use FP16:** Faster than FP32
4. **Reduce steps:** 4 steps is optimal for LCM

### Canny Edges Too Strong/Weak

```bash
# For cluttered images (detect less edges)
--canny_low 150 --canny_high 300

# For low-contrast images (detect more edges)
--canny_low 50 --canny_high 150
```

---

## Hardware Benchmarks

### RTX 3060 (6GB VRAM)

| Configuration | Time/Image | VRAM (A100) | CPU Offload | Notes |
|--------------|------------|-------------|-------------|-------|
| **SSD-1B FP16**  | **~6s**    | 5.3 GB      | **Enabled** | **Recommended - 100x faster than DDIM** |
| SSD-1B FP16 (no offload) | ~25s | 5.3 GB | Disabled | **DRAM paging - 4.2x slower!** |
| SSD-1B FP32  | ~118s      | 10.5 GB     | Required    | Usable but slow |
| SDXL FP16    | ~113s      | 14.3 GB     | Required    | Similar to SSD-1B FP32 |
| SDXL FP32    | CRASH      | 26.5 GB     | N/A         | System crash - too large |

**Critical Lesson**: CPU offloading is NOT "just slower" - it's intelligent memory management that prevents catastrophic DRAM paging. Without offloading, even SSD-1B FP16 (5.3GB) becomes 4.2x slower on 6GB GPU due to memory overflow.

### Google Colab A100 (80GB VRAM)

| Configuration | VRAM Usage | Quality | Notes |
|--------------|------------|---------|-------|
| SSD-1B FP16  | 5.3 GB     | SSIM: 0.436 | Fastest, best for consumer GPU emulation |
| SSD-1B FP32  | 10.5 GB    | SSIM: 0.436 (identical) | No quality gain vs FP16 |
| SDXL FP16    | 14.3 GB    | SSIM: 0.464 | 6.4% better structure than SSD-1B |
| SDXL FP32    | 26.5 GB    | SSIM: 0.464 (identical) | No quality gain vs FP16 |

**All benchmarks used CPU offloading disabled** (ample VRAM available on A100)

---

## Additional Resources

- [README.md](README.md) - Research report and benchmark results
- [BATCH_COMMANDS.md](BATCH_COMMANDS.md) - Detailed batch processing commands (if exists)
- PIE-Bench dataset: [Link to dataset]
- Model weights are auto-downloaded from Hugging Face on first run

---

## License

This project is for research and educational purposes. Model weights and datasets follow their respective licenses.
