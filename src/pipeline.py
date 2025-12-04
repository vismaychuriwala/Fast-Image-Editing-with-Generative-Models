"""
Fast Image Editing Pipeline using SDXL/SSD-1B + LCM-LoRA + ControlNet
"""
import torch
import cv2
import numpy as np
from PIL import Image
from diffusers import (
    StableDiffusionXLControlNetImg2ImgPipeline,
    ControlNetModel,
    LCMScheduler,
    UNet2DConditionModel,
    AutoencoderKL,
)


class FastEditor:
    """
    Fast image editor combining SDXL/SSD-1B, ControlNet (Canny), and LCM-LoRA.

    This pipeline enables 4-step image editing while preserving structure
    through Canny edge conditioning.

    Supports two model options:
    - SDXL: Full quality, higher VRAM (5-6GB)
    - SSD-1B: 50% smaller, 60% faster, better for 6GB VRAM
    """

    # Model configurations
    MODEL_CONFIGS = {
        "sdxl": {
            "base_model": "stabilityai/stable-diffusion-xl-base-1.0",
            "lcm_lora": "latent-consistency/lcm-lora-sdxl",
            "use_full_lcm": False,  # Use LoRA adapter
            "description": "Full SDXL (highest quality, ~6GB VRAM)"
        },
        "ssd-1b": {
            "base_model": "segmind/SSD-1B",
            "lcm_model": "latent-consistency/lcm-ssd-1b",
            "use_full_lcm": True,
            "description": "SSD-1B distilled (50% smaller, 60% faster, ~4GB VRAM)"
        }
    }

    def __init__(self, model_name="sdxl", device="cuda", dtype=torch.float16,
                 enable_cpu_offload=True, use_full_precision=False, use_full_controlnet=False):
        """
        Initialize the FastEditor pipeline.

        Args:
            model_name: Model to use ('sdxl' or 'ssd-1b')
            device: Device to run on ('cuda' or 'cpu')
            dtype: Data type for models (torch.float16 or torch.float32)
            enable_cpu_offload: Enable CPU offloading (default: True)
                               Disable for faster inference if you have 8GB+ VRAM
            use_full_precision: Use float32 for maximum quality (A100 recommended)
                               Overrides dtype parameter when True
            use_full_controlnet: Use full-size ControlNet instead of small variant
                                (Higher quality, needs ~2GB more VRAM)
        """
        if model_name not in self.MODEL_CONFIGS:
            raise ValueError(f"Unknown model: {model_name}. Choose from {list(self.MODEL_CONFIGS.keys())}")

        self.model_name = model_name
        self.device = device
        # Override dtype if full precision requested
        if use_full_precision:
            self.dtype = torch.float32
            print(f"[FastEditor] Full precision mode enabled (fp32)")
        else:
            self.dtype = dtype
        self.enable_cpu_offload = enable_cpu_offload
        self.use_full_controlnet = use_full_controlnet
        self.config = self.MODEL_CONFIGS[model_name]

        print(f"[FastEditor] Initializing with {model_name.upper()}")
        print(f"[FastEditor] {self.config['description']}")
        print(f"[FastEditor] Device: {device}, Dtype: {self.dtype}")
        print(f"[FastEditor] CPU offload: {'enabled' if enable_cpu_offload else 'disabled'}")

        # 1. Load ControlNet (Canny) for structure preservation
        if use_full_controlnet:
            print("[FastEditor] Loading ControlNet (Canny) - FULL SIZE for maximum quality...")
            controlnet_model = "diffusers/controlnet-canny-sdxl-1.0"
        else:
            print("[FastEditor] Loading ControlNet (Canny) - small variant...")
            controlnet_model = "diffusers/controlnet-canny-sdxl-1.0-small"

        self.controlnet = ControlNetModel.from_pretrained(
            controlnet_model,
            torch_dtype=self.dtype
        )
        # Load VAE - use fp16 fix for fp16, original for fp32
        if self.dtype == torch.float32:
            print("[FastEditor] Loading VAE (fp32 for maximum quality)...")
            vae = AutoencoderKL.from_pretrained(
                "stabilityai/sdxl-vae",
                torch_dtype=self.dtype
            )
        else:
            print("[FastEditor] Loading VAE (fp16-fix)...")
            vae = AutoencoderKL.from_pretrained(
                "madebyollin/sdxl-vae-fp16-fix",
                torch_dtype=self.dtype
            )
        
        PipelineClass = StableDiffusionXLControlNetImg2ImgPipeline

        # 2. Load base model with appropriate LCM configuration
        if self.config["use_full_lcm"]:
            # For SSD-1B: Load full LCM UNet model
            print(f"[FastEditor] Loading LCM UNet for {model_name.upper()}...")
            # Use variant only for fp16, omit for fp32
            if self.dtype == torch.float16:
                unet = UNet2DConditionModel.from_pretrained(
                    self.config["lcm_model"],
                    torch_dtype=self.dtype,
                    variant="fp16"
                )
            else:
                unet = UNet2DConditionModel.from_pretrained(
                    self.config["lcm_model"],
                    torch_dtype=self.dtype
                )

            print(f"[FastEditor] Loading {model_name.upper()} base model with LCM UNet...")
            # Use variant only for fp16, omit for fp32
            self.pipe = PipelineClass.from_pretrained(
                self.config["base_model"],
                unet=unet, 
                controlnet=self.controlnet,
                vae=vae,
                torch_dtype=self.dtype,
                variant="fp16" if self.dtype == torch.float16 else None
            ).to(device)
            
            print("[FastEditor] Setting LCM scheduler (Manual Config)...")
            self.pipe.scheduler = LCMScheduler.from_config(
                self.pipe.scheduler.config,
                timestep_spacing="trailing"
            )
        else:
            # === SDXL PATH ===
            # We must load the base model here because the code above was skipped
            # Use variant only for fp16, omit for fp32
            print(f"[FastEditor] Loading {model_name.upper()} Img2Img Pipeline...")
            self.pipe = PipelineClass.from_pretrained(
                self.config["base_model"],
                controlnet=self.controlnet,
                vae=vae,
                torch_dtype=self.dtype,
                variant="fp16" if self.dtype == torch.float16 else None
            ).to(device)
            self.pipe.load_lora_weights(self.config["lcm_lora"])

            # 4. Set LCM scheduler (Trailing spacing)
            print("[FastEditor] Setting LCM scheduler...")
            self.pipe.scheduler = LCMScheduler.from_config(
                self.pipe.scheduler.config,
                timestep_spacing="trailing"
            )

        # 5. Enable memory optimizations
        print("[FastEditor] Enabling memory optimizations...")
        if self.enable_cpu_offload:
            self.pipe.enable_model_cpu_offload()  # Move inactive modules to CPU
            print("[FastEditor]   - CPU offload enabled (saves ~2-3GB VRAM, adds latency)")
        else:
            print("[FastEditor]   - CPU offload disabled (faster, needs more VRAM)")

        # Note: VAE tiling/slicing can cause color artifacts with fp16
        # Only enable if you need to save VRAM and can tolerate slight quality loss
        self.pipe.enable_vae_slicing()        # Process VAE in slices
        # self.pipe.enable_vae_tiling()         # Process VAE in tiles (can cause color banding)

        # Only enable attention slicing if using CPU offload (memory constrained)
        if self.enable_cpu_offload:
            self.pipe.enable_attention_slicing()    # Reduce attention memory
            print("[FastEditor]   - Attention slicing enabled (memory saving)")

        print("[FastEditor] Initialization complete!")

    def preprocess_image(self, image, low_threshold=100, high_threshold=200):
        """
        Convert PIL image to Canny edge map for ControlNet conditioning.

        Args:
            image: PIL Image (RGB)
            low_threshold: Canny low threshold
            high_threshold: Canny high threshold

        Returns:
            PIL Image containing Canny edges (RGB format)
        """
        # Convert to numpy array
        image_np = np.array(image)

        # Convert to grayscale for Canny
        if len(image_np.shape) == 3:
            gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
        else:
            gray = image_np

        # Apply Canny edge detection
        edges = cv2.Canny(gray, low_threshold, high_threshold)

        # Convert to 3-channel RGB (ControlNet expects RGB)
        edges_rgb = np.stack([edges, edges, edges], axis=2)

        return Image.fromarray(edges_rgb)

    def edit(
        self,
        image,
        prompt,
        negative_prompt="",
        strength=0.80,
        num_inference_steps=4,
        guidance_scale=1.5,
        controlnet_conditioning_scale=0.5,
        canny_low_threshold=100,
        canny_high_threshold=200,
        seed=None
    ):
        """
        Edit an image using text prompt with structure preservation.

        Args:
            image: PIL Image (RGB) to edit
            prompt: Text prompt describing desired edit
            negative_prompt: Negative prompt (optional)
            num_inference_steps: Number of diffusion steps (4 for LCM)
            guidance_scale: Guidance scale (1.0-2.0 for LCM, lower than standard)
            controlnet_conditioning_scale: How much to preserve structure (0.0-1.0)
            canny_low_threshold: Canny edge detection low threshold
            canny_high_threshold: Canny edge detection high threshold
            seed: Random seed for reproducibility

        Returns:
            PIL Image (edited result)
        """
        # Set random seed if provided
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        else:
            generator = None

        # Resize to 1024x1024 (SDXL native resolution)
        # Note: Metrics will be computed at 512x512 (PIE-Bench resolution) downstream
        # Resize input to 1024x1024 (Required for SDXL/SSD-1B)
        input_image = image.resize((1024, 1024), Image.LANCZOS)

        # Generate Control Image (Canny Edges)
        control_image = self.preprocess_image(
            input_image,
            low_threshold=canny_low_threshold,
            high_threshold=canny_high_threshold
        )

        # Run the pipeline
        result = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            # CRITICAL FIX: Pass both the original image AND the edges
            image=input_image,           # Source for color/latents
            control_image=control_image, # Source for structure
            strength=strength,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            controlnet_conditioning_scale=controlnet_conditioning_scale,
            generator=generator,
        ).images[0]

        return result

    def clear_memory(self):
        """Clear GPU memory cache."""
        if self.device == "cuda":
            torch.cuda.empty_cache()

    def get_memory_usage(self):
        """
        Get current GPU memory usage.

        Returns:
            dict with allocated and reserved memory in GB
        """
        if self.device == "cuda":
            return {
                "allocated_gb": torch.cuda.memory_allocated() / 1024**3,
                "reserved_gb": torch.cuda.memory_reserved() / 1024**3,
            }
        return {"allocated_gb": 0, "reserved_gb": 0}
