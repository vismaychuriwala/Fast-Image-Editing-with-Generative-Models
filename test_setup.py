"""
Test script to verify installation and GPU availability.
"""
import sys


def test_imports():
    """Test if all required packages are importable."""
    print("Testing package imports...")

    required_packages = [
        ("torch", "PyTorch"),
        ("torchvision", "TorchVision"),
        ("diffusers", "Diffusers"),
        ("transformers", "Transformers"),
        ("peft", "PEFT"),
        ("cv2", "OpenCV"),
        ("PIL", "Pillow"),
        ("torchmetrics", "TorchMetrics"),
        ("numpy", "NumPy"),
        ("tqdm", "tqdm"),
        ("matplotlib", "Matplotlib"),
    ]

    failed = []
    for package, name in required_packages:
        try:
            __import__(package)
            print(f"  ✓ {name}")
        except ImportError:
            print(f"  ✗ {name} - NOT FOUND")
            failed.append(name)

    if failed:
        print(f"\n❌ Missing packages: {', '.join(failed)}")
        print("   Install with: pip install -r requirements.txt")
        return False

    print("\n✓ All packages installed!")
    return True


def test_cuda():
    """Test CUDA availability."""
    import torch

    print("\nTesting CUDA...")
    if not torch.cuda.is_available():
        print("  ✗ CUDA not available")
        print("    This project requires a CUDA-enabled GPU")
        return False

    print(f"  ✓ CUDA available")
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  CUDA version: {torch.version.cuda}")

    # Check memory
    total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"  Total VRAM: {total_memory:.2f} GB")

    if total_memory < 6:
        print(f"  ⚠ Warning: Less than 6GB VRAM. May run out of memory.")

    return True


def test_data():
    """Test if PIE-Bench data is available."""
    import os
    import json

    print("\nTesting data availability...")

    mapping_file = "data/PIE-Bench_v1/mapping_file.json"
    source_dir = "data/PIE-Bench_v1/annotation_images"

    if not os.path.exists(mapping_file):
        print(f"  ✗ Mapping file not found: {mapping_file}")
        return False

    if not os.path.exists(source_dir):
        print(f"  ✗ Source images directory not found: {source_dir}")
        return False

    # Load and check mapping file
    try:
        with open(mapping_file, "r") as f:
            mapping = json.load(f)
        print(f"  ✓ Mapping file loaded: {len(mapping)} entries")
    except Exception as e:
        print(f"  ✗ Error loading mapping file: {e}")
        return False

    # Check a sample image
    sample_id = list(mapping.keys())[0]
    sample_path = os.path.join(source_dir, mapping[sample_id]["image_path"])
    if os.path.exists(sample_path):
        print(f"  ✓ Sample image found: {mapping[sample_id]['image_path']}")
    else:
        print(f"  ✗ Sample image not found: {sample_path}")
        return False

    return True


def test_model_access():
    """Test if we can access Hugging Face models."""
    print("\nTesting model access...")
    print("  Note: First run will download models (~10-15GB)")
    print("  This may take several minutes...")
    return True


def main():
    print("="*60)
    print("FastEditor Setup Test")
    print("="*60)

    all_pass = True

    # Test imports
    if not test_imports():
        all_pass = False

    # Test CUDA
    if not test_cuda():
        all_pass = False

    # Test data
    if not test_data():
        all_pass = False

    # Test model access
    test_model_access()

    print("\n" + "="*60)
    if all_pass:
        print("✓ Setup test PASSED!")
        print("\nYou're ready to go! Try running:")
        print("  python run_single_image.py --image data/PIE-Bench_v1/annotation_images/0_random_140/000000000000.jpg --prompt 'test'")
    else:
        print("✗ Setup test FAILED")
        print("\nPlease fix the issues above before running the pipeline.")
    print("="*60)

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
