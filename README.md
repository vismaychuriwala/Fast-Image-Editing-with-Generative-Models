# Fast Image Editing with SDXL/SSD-1B + LCM-LoRA + ControlNet

Fast image editing pipeline combining:
- **SDXL or SSD-1B** for high-quality generation
  - SDXL: Full quality (~6GB VRAM)
  - SSD-1B: 50% smaller, 60% faster (~4GB VRAM)
- **LCM-LoRA** for 4-step fast inference (10-15x speedup)
- **ControlNet (Canny)** for structure preservation

## Project Structure

```
project/
├── src/
│   ├── __init__.py
│   ├── pipeline.py       # FastEditor class
│   └── metrics.py        # Metrics calculator
├── data/
│   └── PIE-Bench_v1/     # Dataset
├── outputs/              # Generated images
├── results/              # Evaluation results
├── run_single_image.py   # Single image editing
├── evaluate.py           # Batch evaluation
├── requirements.txt
└── README.md
```

## Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

**Note**: Requires CUDA-enabled GPU with at least 6GB VRAM.

## Usage

### 1. Single Image Editing

Edit a single image with a text prompt:

```bash
python run_single_image.py \
    --image data/PIE-Bench_v1/annotation_images/0_random_140/000000000000.jpg \
    --prompt "a slanted rusty mountain bicycle on the road in front of a building" \
    --compute_metrics \
    --show_plot
```

**Arguments:**
- `--image`: Path to input image
- `--prompt`: Editing prompt (text description of desired output)
- `--model`: Model to use - `sdxl` (default) or `ssd-1b`
- `--negative_prompt`: Negative prompt (optional)
- `--steps`: Number of inference steps (default: 4)
- `--guidance`: Guidance scale (default: 1.5 for LCM)
- `--control_scale`: ControlNet conditioning scale (default: 0.5)
- `--seed`: Random seed for reproducibility
- `--compute_metrics`: Compute SSIM, LPIPS, CLIP scores
- `--show_plot`: Save side-by-side comparison

**Example with SSD-1B model (faster, less VRAM):**

```bash
python run_single_image.py \
    --image path/to/image.jpg \
    --prompt "your editing prompt" \
    --model ssd-1b \
    --steps 4 \
    --guidance 1.5 \
    --control_scale 0.7 \
    --seed 42 \
    --compute_metrics
```

### 2. Batch Evaluation

Evaluate edited images against PIE-Bench ground truth:

```bash
python evaluate.py \
    --outputs_dir outputs \
    --results_file results/metrics.csv \
    --summary_file results/summary.json
```

**Arguments:**
- `--mapping_file`: PIE-Bench mapping file (default: `data/PIE-Bench_v1/mapping_file.json`)
- `--source_dir`: Source images directory (default: `data/PIE-Bench_v1/annotation_images`)
- `--outputs_dir`: Directory with edited images
- `--results_file`: Output CSV for detailed metrics
- `--summary_file`: Output JSON for summary statistics

**Output:**
- CSV file with per-image metrics
- JSON file with aggregate statistics (mean, std, median)
- Console summary with overall and per-category metrics

## Metrics

### SSIM (Structural Similarity Index)
- Measures structure preservation vs source image
- Range: 0-1 (higher is better)
- Good values: 0.6-0.8

### LPIPS (Learned Perceptual Image Patch Similarity)
- Measures perceptual distance vs source image
- Range: 0-1+ (lower is better)
- Good values: 0.1-0.3

### CLIP Score
- Measures text-image alignment
- Range: 0-100 (higher is better)
- Good values: 25-35

## Hyperparameter Tuning

### ControlNet Conditioning Scale (`--control_scale`)
- Controls structure preservation strength
- Lower (0.3): More freedom, less structure preservation
- Medium (0.5): Balanced (default)
- Higher (0.7): Strong structure preservation, less editing freedom

### Guidance Scale (`--guidance`)
- Controls prompt adherence
- Lower (1.0): More diverse results
- Medium (1.5): Balanced (default for LCM)
- Higher (2.0): Stronger prompt following (may reduce quality)

### Inference Steps (`--steps`)
- Number of diffusion steps
- 4 steps: Fastest (default for LCM)
- 8 steps: Slightly better quality, 2x slower
- Note: LCM is optimized for 4 steps

## Memory Optimization

For 6GB VRAM, the pipeline uses:
- `enable_model_cpu_offload()`: Offload inactive components to CPU RAM
- `enable_vae_slicing()`: Process VAE in slices
- `enable_attention_slicing()`: Reduce attention memory
- FP16 precision

Expected VRAM usage: ~4.5-5.5GB

## Performance

On RTX 3060 (6GB):
- **Inference time**: ~5-10 seconds per image
- **Speedup vs DDIM 50-step**: 10-15x faster

## Troubleshooting

### Out of Memory (OOM)
1. Close other GPU processes
2. Reduce batch size (process one image at a time)
3. Consider switching to SD 1.5 (see pipeline.py comments)

### Poor editing quality
1. Adjust `--control_scale` (try 0.3-0.7 range)
2. Rephrase prompt to be more descriptive
3. Adjust `--guidance` scale

### Canny edges too strong
1. Adjust thresholds: `--canny_low 50 --canny_high 150`
2. Reduce `--control_scale`

## Citation

If you use this code, please cite:

- **SDXL**: Stable Diffusion XL (Stability AI)
- **LCM**: Latent Consistency Models (Luo et al.)
- **ControlNet**: Adding Conditional Control to Text-to-Image Diffusion Models (Zhang et al.)
- **PIE-Bench**: Editing Benchmark for Text-Guided Image Editing
