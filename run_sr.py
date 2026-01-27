"""
Super-Resolution Runner v3 - Using corrected FlowRestore with trajectory coupling

Key fixes:
1. Restored trajectory coupling: zt_hq = zt_edit + (zt_lq - x_lq)
2. Forward noise at each step: zt_lq = (1-t)*x_lq + t*noise
3. Asymmetric guidance: low for degradation, high for clean
"""

import torch
from PIL import Image
import argparse
import random
import numpy as np
import os
import glob
from tqdm import tqdm

from diffusers import StableDiffusion3Pipeline, FluxPipeline
from FlowRestore_utils import FlowRestoreFLUX, FlowRestoreSD3

# =========================================================================
# Configuration
# =========================================================================

INPUT_DIR = r"Data/Urban 100/X2 Urban100/X2/LOW X2 Urban"
OUTPUT_ROOT = r"outputs/Urban100sr"

# Optimized SR parameters
SR_CONFIG_FLUX = {
    "T_steps": 28,
    "n_avg": 1,
    "degradation_guidance": 1.5,    # Low: don't push hard toward degraded
    "clean_guidance": 5.5,          # High: push hard toward clean
    "n_min": 0,
    "n_max": 24,                    # More steps = more restoration
    "reg_scale": 0.0,
    "adaptive_tolerance": 0.0,
    "min_inference_steps": 10,
    "seed": 42
}

SR_CONFIG_SD3 = {
    "T_steps": 50,
    "n_avg": 1,
    "degradation_guidance": 3.5,
    "clean_guidance": 13.5,         # High for stronger effect
    "n_min": 0,
    "n_max": 40,
    "reg_scale": 0.0,
    "adaptive_tolerance": 0.0,
    "min_inference_steps": 15,
    "seed": 42
}


# =========================================================================
# Helpers
# =========================================================================

def load_model(model_type, device_id):
    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
    print(f"Loading {model_type} on {device}...")

    if model_type == 'SD3':
        pipe = StableDiffusion3Pipeline.from_pretrained(
            "stabilityai/stable-diffusion-3-medium-diffusers",
            torch_dtype=torch.float16,
            text_encoder_3=None,
            tokenizer_3=None
        )
    else:  # FLUX
        pipe = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-dev",
            torch_dtype=torch.float16
        )

    pipe = pipe.to(device)
    return pipe, device


# =========================================================================
# Main
# =========================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device_number", type=int, default=0)
    parser.add_argument("--model_type", type=str, required=True, choices=['SD3', 'FLUX'])
    parser.add_argument("--scale", type=int, default=2)
    parser.add_argument("--content", type=str, default="")
    parser.add_argument("--clean_guidance", type=float, default=None)
    parser.add_argument("--n_max", type=int, default=None)
    parser.add_argument("--input_dir", type=str, default=INPUT_DIR)
    parser.add_argument("--output_dir", type=str, default=OUTPUT_ROOT)

    args = parser.parse_args()

    # Load model
    pipe, device = load_model(args.model_type, args.device_number)
    scheduler = pipe.scheduler

    # Select config
    config = SR_CONFIG_FLUX.copy() if args.model_type == 'FLUX' else SR_CONFIG_SD3.copy()
    
    # Override if specified
    if args.clean_guidance is not None:
        config["clean_guidance"] = args.clean_guidance
    if args.n_max is not None:
        config["n_max"] = args.n_max

    # Set seed
    seed = config["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Find images
    image_paths = glob.glob(os.path.join(args.input_dir, "*.png"))
    if not image_paths:
        image_paths = glob.glob(os.path.join(args.input_dir, "*.jpg"))
    if not image_paths:
        print(f"[Error] No images in {args.input_dir}")
        exit()

    print(f"\n{'='*60}")
    print(f"FlowRestore SR v3 (with trajectory coupling)")
    print(f"{'='*60}")
    print(f"Model: {args.model_type} | Scale: x{args.scale} | Images: {len(image_paths)}")
    print(f"Degradation guidance: {config['degradation_guidance']}")
    print(f"Clean guidance: {config['clean_guidance']}")
    print(f"n_max: {config['n_max']} / T_steps: {config['T_steps']}")
    print(f"{'='*60}\n")

    # Output dir
    exp_name = f"{args.model_type}_x{args.scale}_v3_cg{config['clean_guidance']}"
    final_output_dir = os.path.join(args.output_dir, exp_name)
    os.makedirs(final_output_dir, exist_ok=True)

    # Process
    for image_path in tqdm(image_paths, desc="Processing"):
        try:
            # Load and preprocess
            image_lr = Image.open(image_path).convert("RGB")
            target_w = (image_lr.width * args.scale) // 16 * 16
            target_h = (image_lr.height * args.scale) // 16 * 16
            
            image_upscaled = image_lr.resize((target_w, target_h), Image.BICUBIC)
            image_tensor = pipe.image_processor.preprocess(image_upscaled)
            image_tensor = image_tensor.to(device).half()

            with torch.autocast("cuda"), torch.inference_mode():
                x0_src = pipe.vae.encode(image_tensor).latent_dist.mode()

            shift = getattr(pipe.vae.config, "shift_factor", 0.0) or 0.0
            scale_factor = getattr(pipe.vae.config, "scaling_factor", 1.0)
            x_lq = (x0_src - shift) * scale_factor

            # Run FlowRestore
            if args.model_type == 'FLUX':
                x0_tar = FlowRestoreFLUX(
                    pipe=pipe,
                    scheduler=scheduler,
                    x_lq=x_lq,
                    task="sr",
                    content_prompt=args.content,
                    **{k: v for k, v in config.items() if k != "seed"}
                )
            else:
                x0_tar = FlowRestoreSD3(
                    pipe=pipe,
                    scheduler=scheduler,
                    x_lq=x_lq,
                    task="sr",
                    content_prompt=args.content,
                    **{k: v for k, v in config.items() if k != "seed"}
                )

            # Decode and save
            x0_tar_denorm = (x0_tar / scale_factor) + shift
            with torch.autocast("cuda"), torch.inference_mode():
                image_tar = pipe.vae.decode(x0_tar_denorm, return_dict=False)[0]
            image_tar = pipe.image_processor.postprocess(image_tar)

            save_path = os.path.join(final_output_dir, os.path.basename(image_path))
            image_tar[0].save(save_path)

        except Exception as e:
            print(f"\n[Error] {os.path.basename(image_path)}: {e}")
            import traceback
            traceback.print_exc()
            continue

    print(f"\n✓ Done! Results: {final_output_dir}")