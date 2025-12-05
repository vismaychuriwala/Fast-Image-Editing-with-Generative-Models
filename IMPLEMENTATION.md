# Implementation Guide

This document provides technical details for running the fast image editing pipeline.

For the research report and benchmark results, see [README.md](README.md).

---

## Installation

### Requirements

- Python 3.8+
- CUDA-enabled GPU (6GB+ VRAM recommended)
- ~20GB disk space for models

### Setup

```bash
# Clone repository
git clone https://github.com/vismaychuriwala/Fast-Image-Editing-with-Generative-Models
cd Fast-Image-Editing-with-Generative-Models

# Install dependencies
pip install -r requirements.txt
```

**VRAM Requirements:**
- SSD-1B FP16: ~8.5 GB
- SDXL FP16: ~11.2 GB
- SSD-1B FP32: ~17.3 GB
- SDXL FP32: ~22.7 GB

*Note: CPU offloading allows running on GPUs with lower VRAM (e.g., 6GB GPUs can run all FP16 configurations).*

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

```bash
python run_single_image.py \
    --image path/to/image.jpg \
    --prompt "your edit prompt" \
    --model ssd-1b \
    --show_plot
```

### Batch Processing

```bash
python run_batch.py \
    --model ssd-1b \
    --num_images 50
```

---

## Command-Line Flags

### Key Parameters

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `ssd-1b` | Model choice: `ssd-1b` (faster) or `sdxl` (better quality) |
| `--full_precision` | False | Use FP32 instead of FP16 |
| `--steps` | 4 | Number of inference steps (4 recommended for LCM) |
| `--guidance` | 1.5 | CFG scale (1.5 recommended for LCM) |
| `--strength` | 0.5 | img2img strength (0.5 default, tune per image) |
| `--control_scale` | 0.5 | ControlNet influence (0.3-0.7 range) |
| `--no_cpu_offload` | False | Disable CPU offloading (requires more VRAM) |
| `--seed` | Random | Random seed for reproducibility |

### Usage Examples

```bash
# Use SDXL instead of SSD-1B
python run_single_image.py --image img.jpg --prompt "edit" --model sdxl

# Use FP32 precision
python run_single_image.py --image img.jpg --prompt "edit" --full_precision

# Adjust strength for more aggressive editing
python run_single_image.py --image img.jpg --prompt "edit" --strength 0.7
```

---

## Generating Visualizations

Compare different configurations side-by-side:

```bash
# Compare all 4 methods (SDXL FP16/FP32, SSD-1B FP16/FP32)
python plotting/compare_methods.py 000000000037

# Compare SDXL FP16 vs FP32
python plotting/compare_methods.py 000000000000 --methods sdxl_fp16 sdxl_fp32

# Compare SDXL vs SSD-1B (both FP16)
python plotting/compare_methods.py 000000000086 --methods sdxl_fp16 ssd-1b_fp16
```

Output saved to `figures/comparison_*.png`

---

## Reproducing Benchmark Results

Use the Google Colab notebook [`run_benchmark_colab.ipynb`](run_benchmark_colab.ipynb) to reproduce the 700-image benchmark on PIE-Bench.

**For different configurations:**
- **SDXL**: Change `--model ssd-1b` to `--model sdxl`
- **FP32**: Add `--full_precision` flag
- **Results**: Available at [Benchmark-Results](https://drive.google.com/drive/folders/1pH7OeOgON-G1yqzvCgvAKYoyR65F5vvV?usp=sharing)

---

## Additional Information

- **Full research report**: See [README.md](README.md)
- **Model weights**: Auto-downloaded from Hugging Face on first run
- **PIE-Bench dataset**: Required for benchmark reproduction
