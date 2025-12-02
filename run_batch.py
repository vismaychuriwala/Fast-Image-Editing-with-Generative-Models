"""
Run fast image editing on a batch of PIE-Bench images.

Usage:
    python run_batch.py --num_images 50 --editing_types 0 1 2
"""
import argparse
import os
import json
import time
from PIL import Image
from tqdm import tqdm

from src.pipeline import FastEditor


def load_mapping_file(mapping_path):
    """Load PIE-Bench mapping file."""
    with open(mapping_path, "r") as f:
        mapping = json.load(f)
    return mapping


def safe_join(base_dir, user_path):
    """Safely join paths, preventing directory traversal."""
    # Normalize the user path
    user_path = os.path.normpath(user_path)

    # Ensure no absolute paths or parent directory references
    if os.path.isabs(user_path) or user_path.startswith('..'):
        raise ValueError(f"Invalid path: {user_path}")

    # Join and verify result is within base directory
    full_path = os.path.abspath(os.path.join(base_dir, user_path))
    base_abs = os.path.abspath(base_dir)

    if not full_path.startswith(base_abs):
        raise ValueError(f"Path traversal detected: {user_path}")

    return full_path


def main():
    parser = argparse.ArgumentParser(description="Batch image editing on PIE-Bench")
    parser.add_argument("--mapping_file", type=str,
                        default="data/PIE-Bench_v1/mapping_file.json",
                        help="Path to PIE-Bench mapping file")
    parser.add_argument("--source_dir", type=str,
                        default="data/PIE-Bench_v1/annotation_images",
                        help="Directory containing source images")
    parser.add_argument("--output_dir", type=str, default="outputs",
                        help="Output directory")
    parser.add_argument("--model", type=str, default="sdxl", choices=["sdxl", "ssd-1b"],
                        help="Model to use: sdxl (full quality, ~6GB) or ssd-1b (faster, ~4GB)")
    parser.add_argument("--num_images", type=int, default=None,
                        help="Number of images to process (default: all)")
    parser.add_argument("--editing_types", nargs="+", type=str, default=None,
                        help="Filter by editing type IDs (e.g., 0 1 2)")
    parser.add_argument("--image_ids", nargs="+", type=str, default=None,
                        help="Process specific image IDs")
    parser.add_argument("--steps", type=int, default=4,
                        help="Number of inference steps")
    parser.add_argument("--guidance", type=float, default=1.5,
                        help="Guidance scale")
    parser.add_argument("--control_scale", type=float, default=0.5,
                        help="ControlNet conditioning scale")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed")
    parser.add_argument("--no_cpu_offload", action="store_true",
                        help="Disable CPU offloading (faster but needs more VRAM)")
    parser.add_argument("--skip_existing", action="store_true",
                        help="Skip images that already have outputs")

    args = parser.parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load mapping file
    print(f"\n[1/3] Loading mapping file from {args.mapping_file}")
    mapping = load_mapping_file(args.mapping_file)
    print(f"      Total entries in mapping file: {len(mapping)}")

    # Filter images based on arguments
    selected_entries = []

    if args.image_ids:
        # Process specific image IDs
        print(f"\n[2/3] Filtering by image IDs...")
        for image_id in args.image_ids:
            if image_id in mapping:
                selected_entries.append((image_id, mapping[image_id]))
        print(f"      Selected {len(selected_entries)} images by ID")

    else:
        # Filter by editing type
        if args.editing_types:
            print(f"\n[2/3] Filtering by editing types: {args.editing_types}")
            for image_id, entry in mapping.items():
                if entry.get("editing_type_id") in args.editing_types:
                    selected_entries.append((image_id, entry))
            print(f"      Selected {len(selected_entries)} images by type")
        else:
            selected_entries = list(mapping.items())
            print(f"\n[2/3] Processing all images: {len(selected_entries)}")

        # Limit number of images
        if args.num_images and args.num_images < len(selected_entries):
            selected_entries = selected_entries[:args.num_images]
            print(f"      Limited to first {args.num_images} images")

    if len(selected_entries) == 0:
        print("\n      No images selected. Exiting.")
        return

    # Initialize pipeline
    print(f"\n[3/3] Initializing FastEditor...")
    editor = FastEditor(model_name=args.model, device="cuda",
                       enable_cpu_offload=not args.no_cpu_offload)

    # Display memory usage
    mem = editor.get_memory_usage()
    print(f"      GPU Memory: {mem['allocated_gb']:.2f}GB allocated, {mem['reserved_gb']:.2f}GB reserved")

    # Process images
    print(f"\n      Processing {len(selected_entries)} images...")
    print(f"      Parameters: steps={args.steps}, guidance={args.guidance}, control_scale={args.control_scale}")

    processed = 0
    skipped = 0
    failed = 0
    total_time = 0

    for image_id, entry in tqdm(selected_entries, desc="Editing"):
        try:
            # Get paths (with path traversal protection)
            source_filename = entry["image_path"]
            source_path = safe_join(args.source_dir, source_filename)
            output_path = safe_join(args.output_dir, source_filename)

            # Skip if output exists and skip_existing is set
            if args.skip_existing and os.path.exists(output_path):
                skipped += 1
                continue

            # Check if source exists
            if not os.path.exists(source_path):
                failed += 1
                continue

            # Create output subdirectory if needed
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            # Load image
            source_img = Image.open(source_path).convert("RGB")

            # Get editing prompt
            editing_prompt = entry.get("editing_prompt", "")
            if not editing_prompt:
                failed += 1
                continue

            # Run editing
            start_time = time.time()
            edited_img = editor.edit(
                image=source_img,
                prompt=editing_prompt,
                num_inference_steps=args.steps,
                guidance_scale=args.guidance,
                controlnet_conditioning_scale=args.control_scale,
                seed=args.seed,
            )
            elapsed = time.time() - start_time
            total_time += elapsed

            # Save output
            edited_img.save(output_path)
            processed += 1

            # Clear memory periodically
            if processed % 10 == 0:
                editor.clear_memory()

        except FileNotFoundError as e:
            print(f"\n      File not found for {image_id}: {e}")
            failed += 1
            continue
        except ValueError as e:
            print(f"\n      Invalid path for {image_id}: {e}")
            failed += 1
            continue
        except Exception as e:
            print(f"\n      Error processing {image_id} ({type(e).__name__}): {e}")
            failed += 1
            continue

    # Summary
    print(f"\n{'='*60}")
    print(f"BATCH PROCESSING SUMMARY")
    print(f"{'='*60}")
    print(f"\nProcessed:  {processed} images")
    print(f"Skipped:    {skipped} images")
    print(f"Failed:     {failed} images")
    if processed > 0:
        print(f"\nAverage time per image: {total_time / processed:.2f}s")
        print(f"Total time: {total_time:.2f}s ({total_time / 60:.1f} minutes)")
    else:
        print(f"\n⚠ WARNING: No images were successfully processed!")
        print(f"  Check that:")
        print(f"    - Source images exist at: {args.source_dir}")
        print(f"    - Mapping file is correct: {args.mapping_file}")
        print(f"    - Selected filters match available images")
    print(f"\nOutputs saved to: {args.output_dir}")
    print(f"{'='*60}")

    # Clean up
    editor.clear_memory()

    print("\nDone! Next steps:")
    print(f"  1. Review outputs: ls {args.output_dir}")
    print(f"  2. Run evaluation: python evaluate.py --outputs_dir {args.output_dir}")


if __name__ == "__main__":
    main()
