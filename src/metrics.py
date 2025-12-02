"""
Metrics calculator for image editing evaluation.

Supports:
- SSIM (Structural Similarity Index)
- LPIPS (Learned Perceptual Image Patch Similarity)
- CLIP Score (text-image alignment)
"""
import torch
import numpy as np
from PIL import Image
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.multimodal import CLIPScore


class MetricsCalculator:
    """
    Calculate image quality and editing metrics.

    Metrics:
    - SSIM: Structure preservation (higher is better, 0-1)
    - LPIPS: Perceptual distance (lower is better, 0-1+)
    - CLIP Score: Text-image alignment (higher is better)
    """

    def __init__(self, device="cuda"):
        """
        Initialize metrics calculators.

        Args:
            device: Device to run on ('cuda' or 'cpu')
        """
        self.device = device
        print(f"[MetricsCalculator] Initializing on {device}...")

        # SSIM metric
        self.ssim_metric = StructuralSimilarityIndexMeasure(
            data_range=1.0
        ).to(device)

        # LPIPS metric (using SqueezeNet for speed)
        self.lpips_metric = LearnedPerceptualImagePatchSimilarity(
            net_type='squeeze'
        ).to(device)

        # CLIP Score metric
        self.clip_metric = CLIPScore(
            model_name_or_path="openai/clip-vit-base-patch16"
        ).to(device)

        print("[MetricsCalculator] Initialization complete!")

    def _pil_to_tensor(self, img):
        """
        Convert PIL Image to torch tensor [1, C, H, W] normalized to [0, 1].

        Args:
            img: PIL Image (RGB)

        Returns:
            torch.Tensor of shape [1, 3, H, W]
        """
        img_np = np.array(img).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0)
        return img_tensor.to(self.device)

    def calculate_ssim(self, img1, img2):
        """
        Calculate SSIM between two images.

        Args:
            img1: PIL Image (RGB)
            img2: PIL Image (RGB)

        Returns:
            float: SSIM score (0-1, higher is better)
        """
        assert img1.size == img2.size, f"Images must have same size. Got {img1.size} and {img2.size}"

        img1_tensor = self._pil_to_tensor(img1)
        img2_tensor = self._pil_to_tensor(img2)

        with torch.no_grad():
            score = self.ssim_metric(img1_tensor, img2_tensor)

        return score.item()

    def calculate_lpips(self, img1, img2):
        """
        Calculate LPIPS (perceptual distance) between two images.

        Args:
            img1: PIL Image (RGB)
            img2: PIL Image (RGB)

        Returns:
            float: LPIPS score (0-1+, lower is better)
        """
        assert img1.size == img2.size, f"Images must have same size. Got {img1.size} and {img2.size}"

        img1_tensor = self._pil_to_tensor(img1)
        img2_tensor = self._pil_to_tensor(img2)

        # LPIPS expects input in range [-1, 1]
        img1_tensor = img1_tensor * 2 - 1
        img2_tensor = img2_tensor * 2 - 1

        with torch.no_grad():
            score = self.lpips_metric(img1_tensor, img2_tensor)

        return score.item()

    def calculate_clip_score(self, img, text):
        """
        Calculate CLIP score (text-image alignment).

        Args:
            img: PIL Image (RGB)
            text: Text prompt (string)

        Returns:
            float: CLIP score (higher is better, typically 0-100)
        """
        # Convert image to tensor [3, H, W] in range [0, 255]
        img_np = np.array(img)
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).to(self.device)

        with torch.no_grad():
            score = self.clip_metric(img_tensor, text)

        return score.item()

    def calculate_all_metrics(self, source_img, edited_img, prompt):
        """
        Calculate all metrics for an edited image.

        Args:
            source_img: PIL Image (original)
            edited_img: PIL Image (edited result)
            prompt: Text prompt used for editing

        Returns:
            dict with all metric scores
        """
        metrics = {}

        # Structure preservation (vs source)
        metrics["ssim"] = self.calculate_ssim(source_img, edited_img)

        # Perceptual distance (vs source)
        metrics["lpips"] = self.calculate_lpips(source_img, edited_img)

        # Text-image alignment
        metrics["clip_score"] = self.calculate_clip_score(edited_img, prompt)

        return metrics

    def clear_memory(self):
        """Clear GPU memory cache."""
        if self.device == "cuda":
            torch.cuda.empty_cache()
