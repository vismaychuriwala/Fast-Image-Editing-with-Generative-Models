"""
Run fast image editing on a single image.

Usage:
    python run_single_image.py --image path/to/image.jpg --prompt "a rusty bicycle"
"""
import argparse
import os
import time
from datetime import datetime
from PIL import Image
import matplotlib.pyplot as plt

from src.pipeline import FastEditor
from src.metrics import MetricsCalculator


def main():
    parser = argparse.ArgumentParser(description="Fast image editing on a single image")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--prompt", type=str, required=True, help="Editing prompt")
    parser.add_argument("--model", type=str, default="sdxl", choices=["sdxl", "ssd-1b"],
                        help="Model to use: sdxl (full quality, ~6GB) or ssd-1b (faster, ~4GB)")
    parser.add_argument("--negative_prompt", type=str, default="", help="Negative prompt")
    parser.add_argument("--steps", type=int, default=4, help="Number of inference steps")
    parser.add_argument("--guidance", type=float, default=1.5, help="Guidance scale")
    parser.add_argument("--control_scale", type=float, default=0.5, help="ControlNet conditioning scale")
    parser.add_argument("--canny_low", type=int, default=100, help="Canny low threshold")
    parser.add_argument("--canny_high", type=int, default=200, help="Canny high threshold")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Output directory")
    parser.add_argument("--no_cpu_offload", action="store_true",
                        help="Disable CPU offloading (faster but needs more VRAM)")
    parser.add_argument("--quality_mode", action="store_true",
                        help="Maximum quality mode (fp32, full ControlNet) - A100 recommended")
    parser.add_argument("--full_precision", action="store_true",
                        help="Use fp32 instead of fp16 (better quality, 2x VRAM)")
    parser.add_argument("--full_controlnet", action="store_true",
                        help="Use full-size ControlNet instead of small variant")
    parser.add_argument("--compute_metrics", action="store_true", help="Compute metrics")
    parser.add_argument("--show_plot", action="store_true", help="Show comparison plot")

    args = parser.parse_args()

    # Handle quality mode
    if args.quality_mode:
        args.full_precision = True
        args.full_controlnet = True
        args.no_cpu_offload = True
        print("[Quality Mode] Enabled: fp32 + full ControlNet + no CPU offload")

    # Check if image exists
    if not os.path.exists(args.image):
        print(f"Error: Image not found at {args.image}")
        return

    # Create output directory name based on model and precision
    precision_str = "fp32" if args.full_precision else "fp16"
    model_suffix = f"{args.model}_{precision_str}"

    # Create output directories with organized structure
    edited_dir = os.path.join(args.output_dir, "single", "edited", model_suffix)
    comparisons_dir = os.path.join(args.output_dir, "single", "comparisons", model_suffix)
    os.makedirs(edited_dir, exist_ok=True)
    os.makedirs(comparisons_dir, exist_ok=True)

    # Load image
    print(f"\n[1/4] Loading image from {args.image}")
    source_img = Image.open(args.image).convert("RGB")
    print(f"      Image size: {source_img.size}")

    # Initialize pipeline
    print(f"\n[2/4] Initializing FastEditor...")
    editor = FastEditor(
        model_name=args.model,
        device="cuda",
        enable_cpu_offload=not args.no_cpu_offload,
        use_full_precision=args.full_precision,
        use_full_controlnet=args.full_controlnet
    )

    # Display memory usage
    mem = editor.get_memory_usage()
    print(f"      GPU Memory: {mem['allocated_gb']:.2f}GB allocated, {mem['reserved_gb']:.2f}GB reserved")

    # Run editing
    print(f"\n[3/4] Running image editing...")
    print(f"      Prompt: {args.prompt}")
    print(f"      Steps: {args.steps}, Guidance: {args.guidance}, Control Scale: {args.control_scale}")

    start_time = time.time()
    edited_img = editor.edit(
        image=source_img,
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
        controlnet_conditioning_scale=args.control_scale,
        canny_low_threshold=args.canny_low,
        canny_high_threshold=args.canny_high,
        seed=args.seed,
    )
    elapsed_time = time.time() - start_time

    print(f"      Editing completed in {elapsed_time:.2f} seconds")

    # Display memory usage after editing
    mem = editor.get_memory_usage()
    print(f"      GPU Memory: {mem['allocated_gb']:.2f}GB allocated, {mem['reserved_gb']:.2f}GB reserved")

    # Save output to organized folder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(edited_dir, f"edited_{timestamp}.jpg")
    edited_img.save(output_path)
    print(f"\n      Saved edited image to: {output_path}")

    # Compute metrics if requested
    if args.compute_metrics:
        print(f"\n[4/4] Computing metrics...")
        metrics_calc = MetricsCalculator(device="cuda")

        metrics = metrics_calc.calculate_all_metrics(
            source_img=source_img,
            edited_img=edited_img,
            prompt=args.prompt
        )

        print(f"\n      Metrics:")
        print(f"        SSIM (structure preservation):  {metrics['ssim']:.4f}")
        print(f"        LPIPS (perceptual distance):    {metrics['lpips']:.4f}")
        print(f"        PSNR (signal quality):          {metrics['psnr']:.2f} dB")
        print(f"        MSE (pixel difference):         {metrics['mse']:.6f}")
        print(f"        CLIP Score (text alignment):    {metrics['clip_score']:.2f}")

        # Save metrics to edited folder
        metrics_path = os.path.join(edited_dir, f"metrics_{timestamp}.txt")
        with open(metrics_path, "w") as f:
            f.write(f"Image: {args.image}\n")
            f.write(f"Prompt: {args.prompt}\n")
            f.write(f"Model: {args.model}\n")
            f.write(f"Time: {elapsed_time:.2f}s\n")
            f.write(f"\nMetrics:\n")
            f.write(f"  SSIM:       {metrics['ssim']:.4f}\n")
            f.write(f"  LPIPS:      {metrics['lpips']:.4f}\n")
            f.write(f"  PSNR:       {metrics['psnr']:.2f} dB\n")
            f.write(f"  MSE:        {metrics['mse']:.6f}\n")
            f.write(f"  CLIP Score: {metrics['clip_score']:.2f}\n")
        print(f"      Saved metrics to: {metrics_path}")

        # Always save comparison plot when computing metrics
        print(f"\n      Saving comparison plot...")
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        axes[0].imshow(source_img)
        axes[0].set_title("Source Image")
        axes[0].axis("off")

        axes[1].imshow(edited_img)
        axes[1].set_title(f"Edited Image ({args.model.upper()})\n\"{args.prompt}\"")
        axes[1].axis("off")

        plt.tight_layout()
        plot_path = os.path.join(comparisons_dir, f"comparison_{timestamp}.png")
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        print(f"      Saved comparison plot to: {plot_path}")
        plt.close()

    # Show additional comparison plot if explicitly requested (without metrics)
    elif args.show_plot:
        print(f"\n      Saving comparison plot...")
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        axes[0].imshow(source_img)
        axes[0].set_title("Source Image")
        axes[0].axis("off")

        axes[1].imshow(edited_img)
        axes[1].set_title(f"Edited Image ({args.model.upper()})\n\"{args.prompt}\"")
        axes[1].axis("off")

        plt.tight_layout()
        plot_path = os.path.join(comparisons_dir, f"comparison_{timestamp}.png")
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        print(f"      Saved comparison plot to: {plot_path}")
        plt.close()

    # Clean up
    editor.clear_memory()
    if args.compute_metrics:
        metrics_calc.clear_memory()

    print(f"\nDone!")


if __name__ == "__main__":
    main()
