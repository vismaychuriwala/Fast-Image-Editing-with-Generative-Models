"""
Flexible script to compare editing results across different configurations.
Supports comparing all 4 methods or specific subsets (e.g., FP16 vs FP32).
"""
import os
import sys
import argparse
import json
import matplotlib.pyplot as plt
from PIL import Image


def plot_comparison(image_id, methods=None, data_dir="data/PIE-Bench_v1",
                   outputs_dir="outputs/batch/edited", output_dir="figures"):
    """
    Plot comparison of source image and selected editing methods for a given image ID.

    Args:
        image_id: Image ID (e.g., "000000000000")
        methods: List of method configs to compare. If None, compares all 4.
                 Examples: ["sdxl_fp16", "sdxl_fp32"] or ["sdxl_fp16", "ssd-1b_fp16"]
        data_dir: Path to PIE-Bench dataset
        outputs_dir: Path to batch edited outputs
        output_dir: Directory to save output figures (default: "figures")

    Returns:
        Path to saved figure
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Define all available methods with display names
    all_methods = {
        "sdxl_fp32": "SDXL FP32",
        "sdxl_fp16": "SDXL FP16",
        "ssd-1b_fp32": "SSD-1B FP32",
        "ssd-1b_fp16": "SSD-1B FP16",
    }

    # Use all methods if none specified
    if methods is None:
        methods = list(all_methods.keys())

    # Validate method names
    for method in methods:
        if method not in all_methods:
            print(f"Error: Unknown method '{method}'. Available: {list(all_methods.keys())}")
            return None

    # Load mapping file to get prompt and path
    mapping_file = os.path.join(data_dir, "mapping_file.json")
    with open(mapping_file, 'r') as f:
        mapping = json.load(f)

    # Get image data from mapping (dictionary with image_id as key)
    if image_id not in mapping:
        print(f"Error: Image ID {image_id} not found in mapping file")
        return None

    image_data = mapping[image_id]

    # Get source image path and prompt
    source_path = os.path.join(data_dir, "annotation_images", image_data['image_path'])
    prompt = image_data['editing_prompt']

    # Load source image
    source_img = Image.open(source_path).convert('RGB')

    # Load edited images for selected methods
    edited_images = []
    method_labels = []
    for config in methods:
        edited_path = os.path.join(outputs_dir, config, image_data['image_path'])
        if os.path.exists(edited_path):
            edited_images.append(Image.open(edited_path).convert('RGB'))
            method_labels.append(all_methods[config])
        else:
            print(f"Warning: {edited_path} not found, skipping")

    # Create figure with source + selected methods
    n_images = 1 + len(edited_images)
    fig, axes = plt.subplots(1, n_images, figsize=(4*n_images, 4))

    # Ensure axes is always a list
    if n_images == 1:
        axes = [axes]

    # Plot source image
    axes[0].imshow(source_img)
    axes[0].set_title("Source Image", fontsize=12, fontweight='bold')
    axes[0].axis('off')

    # Plot edited images
    for idx, (img, label) in enumerate(zip(edited_images, method_labels)):
        axes[idx + 1].imshow(img)
        axes[idx + 1].set_title(label, fontsize=12, fontweight='bold')
        axes[idx + 1].axis('off')

    # Add prompt as suptitle
    plt.suptitle(f"Prompt: {prompt}", fontsize=14, y=0.98, fontweight='bold')
    plt.tight_layout()

    # Generate output filename based on methods being compared
    if len(methods) == len(all_methods):
        # All methods - use simple name
        output_filename = f"comparison_all_{image_id}.png"
    else:
        # Subset - encode in filename
        methods_str = "_vs_".join(methods)
        output_filename = f"comparison_{methods_str}_{image_id}.png"

    output_path = os.path.join(output_dir, output_filename)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved comparison to {output_path}")

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Compare editing results across different model configurations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare all 4 methods
  python compare_methods.py 000000000000

  # Compare SDXL FP16 vs FP32
  python compare_methods.py 000000000000 --methods sdxl_fp16 sdxl_fp32

  # Compare SSD-1B FP16 vs FP32
  python compare_methods.py 000000000000 --methods ssd-1b_fp16 ssd-1b_fp32

  # Compare SDXL FP16 vs SSD-1B FP16 (model comparison)
  python compare_methods.py 000000000000 --methods sdxl_fp16 ssd-1b_fp16
        """
    )
    parser.add_argument("image_id", help="Image ID to compare (e.g., 000000000000)")
    parser.add_argument("--methods", nargs="+",
                       help="Methods to compare (default: all 4). Options: sdxl_fp32, sdxl_fp16, ssd-1b_fp32, ssd-1b_fp16")
    parser.add_argument("--output_dir", default="figures",
                       help="Output directory for figures (default: figures)")

    args = parser.parse_args()

    plot_comparison(args.image_id, methods=args.methods, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
