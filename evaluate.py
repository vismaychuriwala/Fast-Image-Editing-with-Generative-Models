"""
Evaluate generated images using PIE-Bench mapping file.

Usage:
    python evaluate.py --outputs_dir outputs --results_file results/metrics.csv
"""
import argparse
import os
import json
import csv
from PIL import Image
from tqdm import tqdm
import numpy as np

from src.metrics import MetricsCalculator


def load_mapping_file(mapping_path):
    """Load PIE-Bench mapping file."""
    with open(mapping_path, "r") as f:
        mapping = json.load(f)
    return mapping


def main():
    parser = argparse.ArgumentParser(description="Evaluate edited images")
    parser.add_argument("--mapping_file", type=str,
                        default="data/PIE-Bench_v1/mapping_file.json",
                        help="Path to PIE-Bench mapping file")
    parser.add_argument("--source_dir", type=str,
                        default="data/PIE-Bench_v1/annotation_images",
                        help="Directory containing source images")
    parser.add_argument("--outputs_dir", type=str, required=True,
                        help="Directory containing edited images (e.g., outputs/batch/edited or outputs/single/edited)")
    parser.add_argument("--results_file", type=str,
                        default="results/metrics.csv",
                        help="Output CSV file for metrics")
    parser.add_argument("--summary_file", type=str,
                        default="results/summary.json",
                        help="Output JSON file for summary statistics")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use for metrics computation")

    args = parser.parse_args()

    # Create results directory
    os.makedirs(os.path.dirname(args.results_file), exist_ok=True)

    # Load mapping file
    print(f"\n[1/3] Loading mapping file from {args.mapping_file}")
    mapping = load_mapping_file(args.mapping_file)
    print(f"      Found {len(mapping)} entries in mapping file")

    # Check outputs directory exists and is accessible
    print(f"\n[2/3] Scanning outputs directory: {args.outputs_dir}")
    if not os.path.exists(args.outputs_dir):
        print(f"Error: Outputs directory not found: {args.outputs_dir}")
        return

    if not os.path.isdir(args.outputs_dir):
        print(f"Error: Not a directory: {args.outputs_dir}")
        return

    try:
        output_files = set(os.listdir(args.outputs_dir))
    except PermissionError:
        print(f"Error: Permission denied reading: {args.outputs_dir}")
        return

    print(f"      Found {len(output_files)} files in outputs directory")

    # Initialize metrics calculator
    print(f"\n[3/3] Computing metrics...")
    metrics_calc = MetricsCalculator(device=args.device)

    # Results storage
    all_results = []
    category_metrics = {}  # Store metrics per editing category

    # Process each image
    processed_count = 0
    skipped_count = 0

    for image_id, entry in tqdm(mapping.items(), desc="Evaluating"):
        # Get paths
        source_filename = entry["image_path"]
        source_path = os.path.join(args.source_dir, source_filename)

        # Check if we have an output for this image
        # Output filename should match source filename
        output_filename = source_filename
        output_path = os.path.join(args.outputs_dir, output_filename)

        # Skip if output doesn't exist
        if not os.path.exists(output_path):
            skipped_count += 1
            continue

        # Skip if source doesn't exist
        if not os.path.exists(source_path):
            skipped_count += 1
            continue

        try:
            # Load images
            source_img = Image.open(source_path).convert("RGB")
            edited_img = Image.open(output_path).convert("RGB")

            # Resize copies to 512x512 for metrics while preserving originals for reporting/saving
            metric_size = (512, 512)
            source_metric = source_img if source_img.size == metric_size else source_img.resize(metric_size, Image.LANCZOS)
            edited_metric = edited_img if edited_img.size == metric_size else edited_img.resize(metric_size, Image.LANCZOS)

            # Get editing information
            editing_prompt = entry.get("editing_prompt", "")
            editing_type = entry.get("editing_type_id", "unknown")

            # Calculate metrics
            metrics = metrics_calc.calculate_all_metrics(
                source_img=source_metric,
                edited_img=edited_metric,
                prompt=editing_prompt
            )

            # Store results
            result = {
                "image_id": image_id,
                "image_path": source_filename,
                "editing_type_id": editing_type,
                "editing_prompt": editing_prompt,
                "ssim": metrics["ssim"],
                "lpips": metrics["lpips"],
                "clip_score": metrics["clip_score"],
                "psnr": metrics["psnr"],
                "mse": metrics["mse"],
                "dino_distance": metrics["dino_distance"],
            }
            all_results.append(result)

            # Accumulate for category statistics
            if editing_type not in category_metrics:
                category_metrics[editing_type] = {
                    "ssim": [],
                    "lpips": [],
                    "clip_score": [],
                    "psnr": [],
                    "mse": [],
                    "dino_distance": [],
                    "count": 0
                }
            category_metrics[editing_type]["ssim"].append(metrics["ssim"])
            category_metrics[editing_type]["lpips"].append(metrics["lpips"])
            category_metrics[editing_type]["clip_score"].append(metrics["clip_score"])
            category_metrics[editing_type]["psnr"].append(metrics["psnr"])
            category_metrics[editing_type]["mse"].append(metrics["mse"])
            category_metrics[editing_type]["dino_distance"].append(metrics["dino_distance"])
            category_metrics[editing_type]["count"] += 1

            processed_count += 1

        except Exception as e:
            print(f"\n      Error processing {image_id}: {e}")
            skipped_count += 1
            continue

    print(f"\n      Processed: {processed_count} images")
    print(f"      Skipped:   {skipped_count} images")

    if processed_count == 0:
        print("\n      No images were processed. Exiting.")
        return

    # Save detailed results to CSV
    print(f"\n[4/4] Saving results...")
    with open(args.results_file, "w", newline="") as f:
        fieldnames = ["image_id", "image_path", "editing_type_id", "editing_prompt",
                      "ssim", "lpips", "clip_score", "psnr", "mse", "dino_distance"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results)
    print(f"      Saved detailed metrics to: {args.results_file}")

    # Calculate and save summary statistics
    summary = {
        "total_images": processed_count,
        "overall": {
            "ssim": {
                "mean": float(np.mean([r["ssim"] for r in all_results])),
                "std": float(np.std([r["ssim"] for r in all_results])),
                "median": float(np.median([r["ssim"] for r in all_results])),
            },
            "lpips": {
                "mean": float(np.mean([r["lpips"] for r in all_results])),
                "std": float(np.std([r["lpips"] for r in all_results])),
                "median": float(np.median([r["lpips"] for r in all_results])),
            },
            "clip_score": {
                "mean": float(np.mean([r["clip_score"] for r in all_results])),
                "std": float(np.std([r["clip_score"] for r in all_results])),
                "median": float(np.median([r["clip_score"] for r in all_results])),
            },
            "psnr": {
                "mean": float(np.mean([r["psnr"] for r in all_results])),
                "std": float(np.std([r["psnr"] for r in all_results])),
                "median": float(np.median([r["psnr"] for r in all_results])),
            },
            "mse": {
                "mean": float(np.mean([r["mse"] for r in all_results])),
                "std": float(np.std([r["mse"] for r in all_results])),
                "median": float(np.median([r["mse"] for r in all_results])),
            },
            "dino_distance": {
                "mean": float(np.mean([r["dino_distance"] for r in all_results])),
                "std": float(np.std([r["dino_distance"] for r in all_results])),
                "median": float(np.median([r["dino_distance"] for r in all_results])),
            },
        },
        "by_category": {}
    }

    # Add per-category statistics
    for category, metrics in category_metrics.items():
        summary["by_category"][category] = {
            "count": metrics["count"],
            "ssim": {
                "mean": float(np.mean(metrics["ssim"])),
                "std": float(np.std(metrics["ssim"])),
            },
            "lpips": {
                "mean": float(np.mean(metrics["lpips"])),
                "std": float(np.std(metrics["lpips"])),
            },
            "clip_score": {
                "mean": float(np.mean(metrics["clip_score"])),
                "std": float(np.std(metrics["clip_score"])),
            },
            "psnr": {
                "mean": float(np.mean(metrics["psnr"])),
                "std": float(np.std(metrics["psnr"])),
            },
            "mse": {
                "mean": float(np.mean(metrics["mse"])),
                "std": float(np.std(metrics["mse"])),
            },
            "dino_distance": {
                "mean": float(np.mean(metrics["dino_distance"])),
                "std": float(np.std(metrics["dino_distance"])),
            },
        }

    # Save summary
    with open(args.summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"      Saved summary statistics to: {args.summary_file}")

    # Print summary to console
    print(f"\n{'='*60}")
    print(f"EVALUATION SUMMARY")
    print(f"{'='*60}")
    print(f"\nTotal Images Evaluated: {processed_count}")
    print(f"\nOverall Metrics:")
    print(f"  SSIM:       {summary['overall']['ssim']['mean']:.4f} ± {summary['overall']['ssim']['std']:.4f}")
    print(f"  LPIPS:      {summary['overall']['lpips']['mean']:.4f} ± {summary['overall']['lpips']['std']:.4f}")
    print(f"  PSNR:       {summary['overall']['psnr']['mean']:.2f} ± {summary['overall']['psnr']['std']:.2f} dB")
    print(f"  MSE:        {summary['overall']['mse']['mean']:.6f} ± {summary['overall']['mse']['std']:.6f}")
    print(f"  CLIP Score: {summary['overall']['clip_score']['mean']:.2f} ± {summary['overall']['clip_score']['std']:.2f}")
    print(f"  DINO Dist.: {summary['overall']['dino_distance']['mean']:.4f} ± {summary['overall']['dino_distance']['std']:.4f}")

    print(f"\nMetrics by Category:")
    for category in sorted(summary["by_category"].keys()):
        cat_data = summary["by_category"][category]
        print(f"\n  Category {category} ({cat_data['count']} images):")
        print(f"    SSIM:       {cat_data['ssim']['mean']:.4f} ± {cat_data['ssim']['std']:.4f}")
        print(f"    LPIPS:      {cat_data['lpips']['mean']:.4f} ± {cat_data['lpips']['std']:.4f}")
        print(f"    PSNR:       {cat_data['psnr']['mean']:.2f} ± {cat_data['psnr']['std']:.2f} dB")
        print(f"    MSE:        {cat_data['mse']['mean']:.6f} ± {cat_data['mse']['std']:.6f}")
        print(f"    CLIP Score: {cat_data['clip_score']['mean']:.2f} ± {cat_data['clip_score']['std']:.2f}")
        print(f"    DINO Dist.: {cat_data['dino_distance']['mean']:.4f} ± {cat_data['dino_distance']['std']:.4f}")

    print(f"\n{'='*60}")
    print("\nDone!")

    # Clean up
    metrics_calc.clear_memory()


if __name__ == "__main__":
    main()
