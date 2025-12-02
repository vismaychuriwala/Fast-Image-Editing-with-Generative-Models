# Quick Start Guide

## Step-by-Step Instructions

### 1. Test Your Setup

```bash
python test_setup.py
```

This will verify:
- All dependencies are installed
- CUDA is available
- PIE-Bench data is accessible
- GPU has sufficient memory

### 2. Run Single Image Test

Test the pipeline on one image:

```bash
python run_single_image.py \
    --image data/PIE-Bench_v1/annotation_images/0_random_140/000000000000.jpg \
    --prompt "a slanted rusty mountain bicycle on the road in front of a building" \
    --compute_metrics
```

**What this does:**
- Loads the image
- Applies Canny edge detection
- Runs SDXL + LCM-LoRA editing
- Saves output to `outputs/`
- Computes SSIM, LPIPS, CLIP scores
- Takes ~5-10 seconds

**Check output:**
```bash
ls outputs/
```

### 3. Run Batch Processing

Process multiple images (start with a small batch):

```bash
# Process 10 images from category 0
python run_batch.py \
    --num_images 10 \
    --editing_types 0 \
    --steps 4 \
    --guidance 1.5 \
    --control_scale 0.5
```

**Time estimate:** ~1-2 minutes for 10 images

### 4. Evaluate Results

Compute metrics for all generated images:

```bash
python evaluate.py \
    --outputs_dir outputs \
    --results_file results/metrics.csv \
    --summary_file results/summary.json
```

**Output files:**
- `results/metrics.csv`: Per-image detailed metrics
- `results/summary.json`: Aggregate statistics
- Console: Summary table

### 5. View Results

```bash
# View CSV
head -20 results/metrics.csv

# View summary
cat results/summary.json
```

## Common Workflows

### Testing Different Hyperparameters

```bash
# Stronger structure preservation
python run_single_image.py \
    --image path/to/image.jpg \
    --prompt "your prompt" \
    --control_scale 0.7

# More editing freedom
python run_single_image.py \
    --image path/to/image.jpg \
    --prompt "your prompt" \
    --control_scale 0.3

# Higher quality (slower)
python run_single_image.py \
    --image path/to/image.jpg \
    --prompt "your prompt" \
    --steps 8
```

### Processing Specific Categories

```bash
# Object replacement (type 0)
python run_batch.py --editing_types 0 --num_images 20

# Style transfer (type 1)
python run_batch.py --editing_types 1 --num_images 20

# Multiple categories
python run_batch.py --editing_types 0 1 2 --num_images 50
```

### Processing Specific Images

```bash
python run_batch.py \
    --image_ids 000000000000 000000000001 000000000002
```

## Troubleshooting

### Out of Memory Error

```bash
# Clear GPU cache and retry
python -c "import torch; torch.cuda.empty_cache()"

# Process fewer images at once
python run_batch.py --num_images 5

# If still failing, consider SD 1.5:
# Edit src/pipeline.py and replace SDXL with SD 1.5
```

### Slow Performance

- **Expected:** 5-10 seconds per image on RTX 3060
- **If slower:** Check GPU utilization with `nvidia-smi`
- **First run:** Model download takes extra time (one-time)

### Poor Results

```bash
# Try different control scale
python run_single_image.py --image ... --prompt "..." --control_scale 0.3
python run_single_image.py --image ... --prompt "..." --control_scale 0.7

# Try different guidance
python run_single_image.py --image ... --prompt "..." --guidance 1.0
python run_single_image.py --image ... --prompt "..." --guidance 2.0

# Adjust Canny thresholds
python run_single_image.py --image ... --prompt "..." --canny_low 50 --canny_high 150
```

## Expected Performance

### RTX 3060 Laptop (6GB VRAM)

| Metric | Value |
|--------|-------|
| Time per image | 5-10 seconds |
| VRAM usage | 4.5-5.5 GB |
| Speedup vs DDIM | 10-15x |
| SSIM (typical) | 0.6-0.8 |
| LPIPS (typical) | 0.1-0.3 |
| CLIP Score (typical) | 25-35 |

### Model Download (First Run)

- SDXL base: ~7GB
- ControlNet: ~1.5GB
- LCM-LoRA: ~100MB
- CLIP model: ~500MB
- **Total:** ~10-15GB

Downloads happen automatically on first run and are cached for future use.

## Next Steps

1. **Test single image** to verify everything works
2. **Run small batch** (10-20 images) to get initial results
3. **Tune hyperparameters** based on results
4. **Scale up** to larger batch (50-100+ images)
5. **Evaluate and analyze** metrics by category

## Need Help?

Check the main [README.md](README.md) for detailed documentation.
