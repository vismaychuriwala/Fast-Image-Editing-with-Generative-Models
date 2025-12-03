"""
Metrics calculator for image editing evaluation.

Supports:
- SSIM (Structural Similarity Index)
- LPIPS (Learned Perceptual Image Patch Similarity)
- CLIP Score (text-image alignment)
- PSNR (Peak Signal-to-Noise Ratio)
- MSE (Mean Squared Error)
- DINO Distance (structural distance via ViT features)
"""
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision.transforms import functional as TF
from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.multimodal import CLIPScore
from torchmetrics.regression import MeanSquaredError


class _VitExtractor:
    """Minimal ViT feature extractor backed by DINO checkpoints."""

    def __init__(self, model_name: str, device: str):
        self.model = torch.hub.load('facebookresearch/dino:main', model_name).to(device)
        self.model.eval()
        self.model_name = model_name
        self.device = device

        self._layers_to_capture = list(range(len(self.model.blocks)))
        self._hook_handlers = []
        self._qkv_outputs = []

    def _register_qkv_hooks(self):
        self._qkv_outputs = []
        for idx, block in enumerate(self.model.blocks):
            if idx in self._layers_to_capture:
                handler = block.attn.qkv.register_forward_hook(self._get_qkv_hook())
                self._hook_handlers.append(handler)

    def _clear_hooks(self):
        for handler in self._hook_handlers:
            handler.remove()
        self._hook_handlers = []

    def _get_qkv_hook(self):
        def _hook(_module, _inp, output):
            self._qkv_outputs.append(output.detach())

        return _hook

    def _run_model(self, input_tensor: torch.Tensor):
        self._register_qkv_hooks()
        with torch.no_grad():
            self.model(input_tensor)
        features = self._qkv_outputs
        self._clear_hooks()
        return features

    def get_qkv_feature_from_input(self, input_img: torch.Tensor):
        return self._run_model(input_img)

    def get_keys_from_input(self, input_img: torch.Tensor, layer_num: int):
        qkv_features = self.get_qkv_feature_from_input(input_img)[layer_num]
        keys = self.get_keys_from_qkv(qkv_features, input_img.shape)
        return keys

    def get_keys_self_sim_from_input(self, input_img: torch.Tensor, layer_num: int):
        keys = self.get_keys_from_input(input_img, layer_num=layer_num)
        heads, tokens, dim = keys.shape
        concatenated = keys.transpose(0, 1).reshape(tokens, heads * dim)
        ssim_map = self.attn_cosine_sim(concatenated[None, None, ...])
        return ssim_map

    def attn_cosine_sim(self, x: torch.Tensor, eps: float = 1e-8):
        x = x[0]
        norm = x.norm(dim=2, keepdim=True)
        factor = torch.clamp(norm @ norm.permute(0, 2, 1), min=eps)
        sim_matrix = (x @ x.permute(0, 2, 1)) / factor
        return sim_matrix

    def get_patch_size(self) -> int:
        return 8 if "8" in self.model_name else 16

    def get_head_num(self) -> int:
        if "dino" in self.model_name:
            return 6 if "s" in self.model_name else 12
        return 6 if "small" in self.model_name else 12

    def get_embedding_dim(self) -> int:
        if "dino" in self.model_name:
            return 384 if "s" in self.model_name else 768
        return 384 if "small" in self.model_name else 768

    def get_patch_num(self, input_shape):
        _, _, height, width = input_shape
        patch = self.get_patch_size()
        return 1 + (height // patch) * (width // patch)

    def get_qkv(self, qkv_tensor: torch.Tensor, input_shape):
        patch_num = self.get_patch_num(input_shape)
        head_num = self.get_head_num()
        embed_dim = self.get_embedding_dim()
        return qkv_tensor.reshape(patch_num, 3, head_num, embed_dim // head_num).permute(1, 2, 0, 3)

    def get_keys_from_qkv(self, qkv_tensor: torch.Tensor, input_shape):
        qkv = self.get_qkv(qkv_tensor, input_shape)
        return qkv[1]


class DinoDistanceMetric:
    """DINO-based structural distance metric used by PIE-Bench baselines."""

    def __init__(self, device: str, model_name: str = "dino_vitb8", resize_to: int = 224, layer: int = 11):
        self.device = device
        self.layer = layer
        self.extractor = _VitExtractor(model_name=model_name, device=device)
        self.resize = transforms.Resize(resize_to, antialias=True)
        self.normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

    def _to_tensor(self, img: Image.Image) -> torch.Tensor:
        if isinstance(img, torch.Tensor):
            tensor = img.detach().clone()
            if tensor.ndim == 4:
                tensor = tensor.squeeze(0)
            tensor = tensor.float()
            if tensor.max() > 1.0:
                tensor = tensor / 255.0
        else:
            tensor = TF.pil_to_tensor(img).float() / 255.0
        tensor = self.resize(tensor)
        tensor = self.normalize(tensor)
        return tensor.to(self.device)

    def calculate_distance(self, source_img: Image.Image, edited_img: Image.Image) -> float:
        src_tensor = self._to_tensor(source_img).unsqueeze(0)
        edt_tensor = self._to_tensor(edited_img).unsqueeze(0)

        with torch.no_grad():
            target_sim = self.extractor.get_keys_self_sim_from_input(src_tensor, layer_num=self.layer)
            edited_sim = self.extractor.get_keys_self_sim_from_input(edt_tensor, layer_num=self.layer)
            distance = F.mse_loss(edited_sim, target_sim)

        return float(distance.item())


class MetricsCalculator:
    """
    Calculate image quality and editing metrics.

    Metrics:
    - SSIM: Structure preservation (higher is better, 0-1)
    - LPIPS: Perceptual distance (lower is better, 0-1+)
    - CLIP Score: Text-image alignment (higher is better)
    - PSNR: Peak Signal-to-Noise Ratio (higher is better, dB)
    - MSE: Mean Squared Error (lower is better, 0-1)
    - DINO Distance: Structural distance between ViT features (lower is better)
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

        # PSNR metric
        self.psnr_metric = PeakSignalNoiseRatio(
            data_range=1.0
        ).to(device)

        # MSE metric
        self.mse_metric = MeanSquaredError().to(device)

        # DINO structure distance
        self.dino_metric = DinoDistanceMetric(device=device)

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
            img1: PIL Image (RGB) - source image
            img2: PIL Image (RGB) - edited image

        Returns:
            float: SSIM score (0-1, higher is better)
        """
        # Resize both images to PIE-Bench resolution (512x512) for fair comparison
        target_size = (512, 512)
        if img1.size != target_size:
            img1 = img1.resize(target_size, Image.LANCZOS)
        if img2.size != target_size:
            img2 = img2.resize(target_size, Image.LANCZOS)

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
        # Resize both images to PIE-Bench resolution (512x512) for fair comparison
        target_size = (512, 512)
        if img1.size != target_size:
            img1 = img1.resize(target_size, Image.LANCZOS)
        if img2.size != target_size:
            img2 = img2.resize(target_size, Image.LANCZOS)

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

    def calculate_psnr(self, img1, img2):
        """
        Calculate PSNR (Peak Signal-to-Noise Ratio) between two images.

        Args:
            img1: PIL Image (RGB)
            img2: PIL Image (RGB)

        Returns:
            float: PSNR score (higher is better, in dB)
        """

        # Resize both images to PIE-Bench resolution (512x512) for fair comparison
        target_size = (512, 512)
        if img1.size != target_size:
            img1 = img1.resize(target_size, Image.LANCZOS)
        if img2.size != target_size:
            img2 = img2.resize(target_size, Image.LANCZOS)

        img1_tensor = self._pil_to_tensor(img1)
        img2_tensor = self._pil_to_tensor(img2)

        with torch.no_grad():
            score = self.psnr_metric(img1_tensor, img2_tensor)

        return score.item()

    def calculate_mse(self, img1, img2):
        """
        Calculate MSE (Mean Squared Error) between two images.

        Args:
            img1: PIL Image (RGB)
            img2: PIL Image (RGB)

        Returns:
            float: MSE score (lower is better, 0-1)
        """

        # Resize both images to PIE-Bench resolution (512x512) for fair comparison
        target_size = (512, 512)
        if img1.size != target_size:
            img1 = img1.resize(target_size, Image.LANCZOS)
        if img2.size != target_size:
            img2 = img2.resize(target_size, Image.LANCZOS)

        img1_tensor = self._pil_to_tensor(img1)
        img2_tensor = self._pil_to_tensor(img2)

        # Flatten tensors for MSE calculation
        img1_flat = img1_tensor.reshape(-1)
        img2_flat = img2_tensor.reshape(-1)

        with torch.no_grad():
            score = self.mse_metric(img1_flat, img2_flat)

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

        # PSNR (signal quality)
        metrics["psnr"] = self.calculate_psnr(source_img, edited_img)

        # MSE (pixel-level difference)
        metrics["mse"] = self.calculate_mse(source_img, edited_img)

        # DINO structural distance
        metrics["dino_distance"] = self.dino_metric.calculate_distance(source_img, edited_img)

        return metrics

    def clear_memory(self):
        """Clear GPU memory cache."""
        if self.device == "cuda":
            torch.cuda.empty_cache()
