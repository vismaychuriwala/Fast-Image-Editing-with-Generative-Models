"""
Fast Image Editing Pipeline using SDXL/SSD-1B + LCM-LoRA + ControlNet
"""
import torch
import cv2
import numpy as np
from PIL import Image
from diffusers import (
    StableDiffusionXLControlNetPipeline,
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
            "controlnet": "diffusers/controlnet-canny-sdxl-1.0",
            "lcm_lora": "latent-consistency/lcm-lora-sdxl",
            "vae_fix": "madebyollin/sdxl-vae-fp16-fix", # Fix for fp16 color artifacts
            "use_full_lcm": False,  # Use LoRA adapter
            "description": "Full SDXL (highest quality, ~6GB VRAM)"
        },
        "ssd-1b": {
            "base_model": "segmind/SSD-1B",
            "controlnet": "diffusers/controlnet-canny-sdxl-1.0",
            "lcm_model": "latent-consistency/lcm-ssd-1b",
            "vae_fix": "madebyollin/sdxl-vae-fp16-fix", # <--- ADD THIS
            "use_full_lcm": True,
            "description": "SSD-1B distilled (50% smaller, 60% faster, ~4GB VRAM)"
        }
    }

    def __init__(self, model_name="sdxl", device="cuda", dtype=torch.float16,
                 enable_cpu_offload=True):
        """
        Initialize the FastEditor pipeline.

        Args:
            model_name: Model to use ('sdxl' or 'ssd-1b')
            device: Device to run on ('cuda' or 'cpu')
            dtype: Data type for models (torch.float16 or torch.float32)
            enable_cpu_offload: Enable CPU offloading (default: True)
                               Disable for faster inference if you have 8GB+ VRAM
        """
        if model_name not in self.MODEL_CONFIGS:
            raise ValueError(f"Unknown model: {model_name}. Choose from {list(self.MODEL_CONFIGS.keys())}")

        self.model_name = model_name
        self.device = device
        self.dtype = dtype
        self.enable_cpu_offload = enable_cpu_offload
        self.config = self.MODEL_CONFIGS[model_name]

        print(f"[FastEditor] Initializing with {model_name.upper()}")
        print(f"[FastEditor] {self.config['description']}")
        print(f"[FastEditor] Device: {device}, Dtype: {dtype}")
        print(f"[FastEditor] CPU offload: {'enabled' if enable_cpu_offload else 'disabled'}")

        # 1. Load ControlNet (Canny) for structure preservation
        print("[FastEditor] Loading ControlNet (Canny)...")
        self.controlnet = ControlNetModel.from_pretrained(
            self.config["controlnet"],
            torch_dtype=dtype
        )
        # Load VAE fix if specified (Corrected logic to apply to BOTH paths)
        vae = None
        if "vae_fix" in self.config:
            print(f"[FastEditor] Loading VAE fix: {self.config['vae_fix']}...")
            vae = AutoencoderKL.from_pretrained(
                self.config["vae_fix"],
                torch_dtype=dtype,
                force_upcast=True
            )

        # 2. Load base model with appropriate LCM configuration
        if self.config["use_full_lcm"]:
            # For SSD-1B: Load full LCM UNet model
            print(f"[FastEditor] Loading LCM UNet for {model_name.upper()}...")
            unet = UNet2DConditionModel.from_pretrained(
                self.config["lcm_model"],
                torch_dtype=dtype,
                variant="fp16"
            )

            print(f"[FastEditor] Loading {model_name.upper()} base model with LCM UNet...")
            self.pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
                self.config["base_model"],
                unet=unet,
                controlnet=self.controlnet,
                vae=vae,  # <--- PASS VAE HERE
                torch_dtype=dtype,
                variant="fp16"
            ).to(device)
            
            print("[FastEditor] Setting LCM scheduler (Manual Config)...")
            self.pipe.scheduler = LCMScheduler.from_config(
                self.pipe.scheduler.config,
                timestep_spacing="trailing"  # <--- CRITICAL for SSD-1B LCM
            )
        else:
            # === SDXL PATH ===
            # We must load the base model here because the code above was skipped
            print(f"[FastEditor] Loading {model_name.upper()} base model...")
            self.pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
                self.config["base_model"],
                controlnet=self.controlnet,
                vae=vae,  # Pass the global VAE here
                torch_dtype=dtype,
            ).to(device)

            # 3. Load LCM-LoRA
            print("[FastEditor] Loading LCM-LoRA weights...")
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

        self.pipe.enable_vae_slicing()        # Process VAE in slices
        self.pipe.enable_attention_slicing()  # Reduce attention memory

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

        # Resize to 512x512 (PIE-Bench resolution)
        image = image.resize((512, 512), Image.LANCZOS)

        # Generate Canny edge map
        control_image = self.preprocess_image(
            image,
            low_threshold=canny_low_threshold,
            high_threshold=canny_high_threshold
        )

        # Run the pipeline
        result = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=control_image,
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
