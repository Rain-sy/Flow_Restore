import torch
import torch.nn.functional as F
from PIL import Image
import argparse
import random 
import numpy as np
import os
import glob
from tqdm import tqdm

# --- 导入 Diffusers Pipelines ---
from diffusers import StableDiffusion3Pipeline, FluxPipeline

# --- 导入 Restoration Utils ---
from FlowRestore_utils import FlowRestoreSD3, FlowRestoreFLUX

# =========================================================================
# 1. 路径设置 (硬编码)
# =========================================================================
INPUT_DIR = r"Data/Urban 100/X2 Urban100/X2/LOW X2 Urban"
OUTPUT_ROOT = r"outputs/Urban100sr"

# =========================================================================
# 2. 模型参数设置 (直接在这里修改)
# =========================================================================
SR_CONFIG = {
    # 采样步数 (FLUX 建议 28-50, SD3 建议 50)
    "T_steps": 50,          
    "n_avg": 1,             
    
    # Guidance Strength (越大越强，SR 推荐 2.0 - 2.5)
    "deg_scale": 2.0,       # 针对低清图的去噪力度
    "clean_scale": 2.0,     # 针对高清图的引导力度
    
    # Intervention Range (介入范围)
    "n_min": 0,             # 从第 0 步开始
    "n_max": 24,            # 到第 24 步结束 (通常设为 T_steps 的一半)
    
    # Regularization (数据一致性约束，SR 核心参数，建议 200-500)
    "reg_scale": 200.0,     
    
    "seed": 42              
}

# Prompt 设置
PROMPTS = {
    "degradation": "a low resolution, blurry, low quality image",
    "clean": "a high resolution, detailed, urban image",
    "negative": "blur, haze, distortion, low quality"
}

# =========================================================================
# 辅助函数
# =========================================================================

def get_sr_degradation_fn(scale_factor):
    """构造 SR 专用的退化函数 (Downsample -> Upscale)"""
    def func(z0_hat, target_size=None):
        # 1. Downsample (Area interpolation)
        z_down = F.interpolate(z0_hat, scale_factor=1/scale_factor, mode='area')
        # 2. Upscale back (Bicubic) - 保持尺寸一致以计算 MSE
        z_reup = F.interpolate(z_down, size=(z0_hat.shape[2], z0_hat.shape[3]), mode='bicubic')
        return z_reup
    return func

def load_model(model_type, device_id):
    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
    print(f"Loading {model_type} on device {device}...")

    if model_type == 'SD3':
        pipe = StableDiffusion3Pipeline.from_pretrained(
            "stabilityai/stable-diffusion-3-medium-diffusers", 
            torch_dtype=torch.float16,
            text_encoder_3=None, 
            tokenizer_3=None
        )
    elif model_type == 'FLUX':
        pipe = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-dev", 
            torch_dtype=torch.float16
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    pipe = pipe.to(device)
    return pipe, device

# =========================================================================
# 主程序
# =========================================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--device_number", type=int, default=0, help="gpu id")
    parser.add_argument("--model_type", type=str, required=True, choices=['SD3', 'FLUX'], 
                        help="Choose model: SD3 or FLUX")
    parser.add_argument("--scale", type=int, default=2, help="Super Resolution Scale")
    
    args = parser.parse_args()

    # 1. 准备模型
    pipe, device = load_model(args.model_type, args.device_number)
    scheduler = pipe.scheduler

    # 2. 设置种子
    seed = SR_CONFIG["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # 3. 准备文件路径
    image_paths = glob.glob(os.path.join(INPUT_DIR, "*.png"))
    # Windows 路径兼容尝试
    if not image_paths:
        image_paths = glob.glob(os.path.join(INPUT_DIR.replace("/", "\\"), "*.png"))
    
    if not image_paths:
        print(f"[Error] No images found in {INPUT_DIR}")
        exit()
    
    print(f"Found {len(image_paths)} images. Mode: {args.model_type}. Scale: x{args.scale}")

    # 4. 自动确定输出目录 (关键步骤)
    # 结果将保存到 outputs/Urban100sr/SD3 或 outputs/Urban100sr/FLUX
    final_output_dir = os.path.join(OUTPUT_ROOT, args.model_type)
    os.makedirs(final_output_dir, exist_ok=True)
    print(f"Saving results to: {final_output_dir}")

    # 5. 构造退化函数
    sr_degradation_fn = get_sr_degradation_fn(args.scale)

    # 6. 开始处理
    for image_src_path in tqdm(image_paths, desc=f"Running {args.model_type}"):
        
        # --- A. 图像预处理 ---
        image_lr = Image.open(image_src_path).convert("RGB")
        
        target_w = image_lr.width * args.scale
        target_h = image_lr.height * args.scale
        
        # 调整为 16 的倍数
        target_w = target_w - (target_w % 16)
        target_h = target_h - (target_h % 16)
        
        # Bicubic 上采样到目标尺寸
        image_upscaled = image_lr.resize((target_w, target_h), Image.BICUBIC)
        
        # 转 Latent
        image_tensor = pipe.image_processor.preprocess(image_upscaled)
        image_tensor = image_tensor.to(device).half()
        
        with torch.autocast("cuda"), torch.inference_mode():
            dist = pipe.vae.encode(image_tensor).latent_dist
            x0_src_denorm = dist.mode()
        
        # VAE Scaling 处理
        shift = getattr(pipe.vae.config, "shift_factor", 0.0)
        scale = getattr(pipe.vae.config, "scaling_factor", 1.0)
        if shift is None: shift = 0.0
        
        x0_src = (x0_src_denorm - shift) * scale
        x0_src = x0_src.to(device)

        # --- B. Run Restoration ---
        # 准备通用参数
        common_params = {
            "pipe": pipe,
            "scheduler": scheduler,
            "x_lq": x0_src,
            "degradation_prompt": PROMPTS["degradation"],
            "clean_prompt": PROMPTS["clean"],
            "T_steps": SR_CONFIG["T_steps"],
            "n_avg": SR_CONFIG["n_avg"],
            "degradation_guidance_scale": SR_CONFIG["deg_scale"],
            "clean_guidance_scale": SR_CONFIG["clean_scale"],
            "n_min": SR_CONFIG["n_min"],
            "n_max": SR_CONFIG["n_max"],
            "reg_scale": SR_CONFIG["reg_scale"],
            "degradation_fn": sr_degradation_fn
        }

        if args.model_type == 'SD3':
            # SD3 需要 negative_prompt
            x0_tar = FlowRestoreSD3(
                negative_prompt=PROMPTS["negative"],
                **common_params
            )
            
        elif args.model_type == 'FLUX':
            # FLUX 不接受 negative_prompt
            x0_tar = FlowRestoreFLUX(
                **common_params
            )

        # --- C. Decode & Save ---
        x0_tar_denorm = (x0_tar / scale) + shift
        
        with torch.autocast("cuda"), torch.inference_mode():
            image_tar = pipe.vae.decode(x0_tar_denorm, return_dict=False)[0]
        
        image_tar = pipe.image_processor.postprocess(image_tar)
        
        # 保存 (直接覆盖同名文件)
        src_filename = os.path.basename(image_src_path)
        save_path = os.path.join(final_output_dir, src_filename)
        image_tar[0].save(save_path)

    print(f"\n{args.model_type} Super-Resolution tasks finished.")