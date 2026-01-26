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

# --- 导入修复后的 Restoration Utils ---
from FlowRestore_utils import FlowRestoreSR_SD3, FlowRestoreSR_FLUX

# =========================================================================
# 1. 路径设置 (硬编码)
# =========================================================================
INPUT_DIR = r"Data/Urban 100/X2 Urban100/X2/LOW X2 Urban"
OUTPUT_ROOT = r"outputs/Urban100sr"

# =========================================================================
# 2. 模型参数设置 (针对SR优化后的推荐值)
# =========================================================================
SR_CONFIG = {
    # 采样步数
    "T_steps": 30,          # FLUX: 28-30, SD3: 40-50
    "n_avg": 1,             # SR不需要多次平均
    
    # Guidance Strength (新版只需要一个guidance)
    "guidance_scale": 3.5,  # 推荐 3.0-4.0 (有内容prompt) 或 2.0-3.0 (空prompt)
    
    # Intervention Range
    "n_min": 0,             
    "n_max": 22,            # FLUX建议22, SD3建议35
    
    # Regularization (SR的核心参数！)
    "reg_scale": 300.0,     # 推荐范围: 200-500
                            # 越大越保守(接近bicubic)
                            # 越小越锐化(可能产生artifacts)
    
    # 自适应停止
    "adaptive_tolerance": 1e-4,  # 0表示关闭
    "min_inference_steps": 10,
    
    "seed": 42              
}

# Prompt 策略选择
PROMPT_MODE = "content"  # 可选: "content", "null", "generic"

# 不同策略的Prompt配置
PROMPT_CONFIGS = {
    # 策略1: 内容描述 (推荐 - 效果最好)
    "content": {
        "use_null_prompt": False,
        "content_description": "urban buildings and streets"  # 根据Urban100修改
    },
    
    # 策略2: 空Prompt (最保守 - 不引入语义)
    "null": {
        "use_null_prompt": True,
        "content_description": ""
    },
    
    # 策略3: 通用高质量描述 (中庸)
    "generic": {
        "use_null_prompt": False,
        "content_description": ""  # 会自动用默认的"high resolution..."
    }
}

# =========================================================================
# 辅助函数
# =========================================================================

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

def adjust_sr_config_for_model(config, model_type):
    """根据模型类型自动调整参数"""
    adjusted = config.copy()
    
    if model_type == 'FLUX':
        adjusted["T_steps"] = min(config["T_steps"], 30)
        adjusted["n_max"] = min(config["n_max"], 22)
    elif model_type == 'SD3':
        adjusted["T_steps"] = max(config["T_steps"], 40)
        adjusted["n_max"] = min(config["n_max"], 35)
    
    return adjusted

# =========================================================================
# 主程序
# =========================================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--device_number", type=int, default=0, help="gpu id")
    parser.add_argument("--model_type", type=str, required=True, choices=['SD3', 'FLUX'], 
                        help="Choose model: SD3 or FLUX")
    parser.add_argument("--scale", type=int, default=2, help="Super Resolution Scale")
    parser.add_argument("--prompt_mode", type=str, default=PROMPT_MODE, 
                        choices=['content', 'null', 'generic'],
                        help="Prompt strategy")
    
    args = parser.parse_args()

    # 1. 准备模型
    pipe, device = load_model(args.model_type, args.device_number)
    scheduler = pipe.scheduler

    # 2. 自动调整配置
    config = adjust_sr_config_for_model(SR_CONFIG, args.model_type)
    
    # 3. 设置种子
    seed = config["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # 4. 准备文件路径
    image_paths = glob.glob(os.path.join(INPUT_DIR, "*.png"))
    if not image_paths:
        image_paths = glob.glob(os.path.join(INPUT_DIR.replace("/", "\\"), "*.png"))
    
    if not image_paths:
        print(f"[Error] No images found in {INPUT_DIR}")
        exit()
    
    print(f"Found {len(image_paths)} images.")
    print(f"Model: {args.model_type} | Scale: x{args.scale} | Prompt Mode: {args.prompt_mode}")
    print(f"Config: T_steps={config['T_steps']}, guidance={config['guidance_scale']}, "
          f"reg={config['reg_scale']}, n_max={config['n_max']}")

    # 5. 自动确定输出目录
    # 格式: outputs/Urban100sr/{model_type}_x{scale}_{prompt_mode}_reg{reg_scale}
    exp_name = f"{args.model_type}_x{args.scale}_{args.prompt_mode}_reg{int(config['reg_scale'])}"
    final_output_dir = os.path.join(OUTPUT_ROOT, exp_name)
    os.makedirs(final_output_dir, exist_ok=True)
    print(f"Saving results to: {final_output_dir}\n")

    # 6. 获取Prompt配置
    prompt_config = PROMPT_CONFIGS[args.prompt_mode]

    # 7. 开始处理
    for image_src_path in tqdm(image_paths, desc=f"Running {args.model_type}"):
        
        # --- A. 图像预处理 ---
        image_lr = Image.open(image_src_path).convert("RGB")
        
        target_w = image_lr.width * args.scale
        target_h = image_lr.height * args.scale
        
        # 调整为 16 的倍数
        target_w = target_w - (target_w % 16)
        target_h = target_h - (target_h % 16)
        
        # Bicubic 上采样到目标尺寸 (作为LR输入)
        image_upscaled = image_lr.resize((target_w, target_h), Image.BICUBIC)
        
        # 转 Latent
        image_tensor = pipe.image_processor.preprocess(image_upscaled)
        image_tensor = image_tensor.to(device).half()
        
        with torch.autocast("cuda"), torch.inference_mode():
            dist = pipe.vae.encode(image_tensor).latent_dist
            x0_src_denorm = dist.mode()
        
        # VAE Scaling 处理
        shift = getattr(pipe.vae.config, "shift_factor", 0.0)
        scale_factor = getattr(pipe.vae.config, "scaling_factor", 1.0)
        if shift is None: shift = 0.0
        
        x_lq = (x0_src_denorm - shift) * scale_factor
        x_lq = x_lq.to(device)

        # --- B. Run Restoration (使用新版API) ---
        
        # 准备通用参数
        common_params = {
            "pipe": pipe,
            "scheduler": scheduler,
            "x_lq": x_lq,
            "content_prompt": prompt_config["content_description"],
            "T_steps": config["T_steps"],
            "n_avg": config["n_avg"],
            "guidance_scale": config["guidance_scale"],
            "n_min": config["n_min"],
            "n_max": config["n_max"],
            "reg_scale": config["reg_scale"],
            "adaptive_tolerance": config.get("adaptive_tolerance", 0.0),
            "min_inference_steps": config.get("min_inference_steps", 10),
            "use_null_prompt": prompt_config["use_null_prompt"]
        }

        try:
            if args.model_type == 'SD3':
                x0_tar = FlowRestoreSR_SD3(**common_params)
                
            elif args.model_type == 'FLUX':
                x0_tar = FlowRestoreSR_FLUX(**common_params)

            # --- C. Decode & Save ---
            x0_tar_denorm = (x0_tar / scale_factor) + shift
            
            with torch.autocast("cuda"), torch.inference_mode():
                image_tar = pipe.vae.decode(x0_tar_denorm, return_dict=False)[0]
            
            image_tar = pipe.image_processor.postprocess(image_tar)
            
            # 保存
            src_filename = os.path.basename(image_src_path)
            save_path = os.path.join(final_output_dir, src_filename)
            image_tar[0].save(save_path)
            
        except Exception as e:
            print(f"\n[Error] Failed on {os.path.basename(image_src_path)}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue

    print(f"\n✓ {args.model_type} Super-Resolution finished!")
    print(f"Results saved to: {final_output_dir}")
    
    # 可选: 保存配置信息
    config_save_path = os.path.join(final_output_dir, "config.txt")
    with open(config_save_path, "w") as f:
        f.write(f"Model: {args.model_type}\n")
        f.write(f"Scale: x{args.scale}\n")
        f.write(f"Prompt Mode: {args.prompt_mode}\n")
        f.write(f"Content Prompt: {prompt_config['content_description']}\n")
        f.write(f"Use Null Prompt: {prompt_config['use_null_prompt']}\n")
        f.write(f"\nConfig:\n")
        for k, v in config.items():
            f.write(f"  {k}: {v}\n")
    print(f"Config saved to: {config_save_path}")