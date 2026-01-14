import torch
from diffusers import FluxPipeline
from PIL import Image
import argparse
import random 
import numpy as np
import yaml
import os
# 导入 FLUX 的工具函数
from FlowEdit_utils import FlowEditFLUX

def load_yaml(path):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.load(f, Loader=yaml.FullLoader)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--device_number", type=int, default=0, help="GPU device ID")
    parser.add_argument("--exp_yaml", type=str, default="sidd_denoise.yaml", help="配置文件路径")

    args = parser.parse_args()
    
    device_id = args.device_number
    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")

    # 1. 加载配置
    exp_configs = load_yaml(args.exp_yaml)

    print(f"🚀 Initializing FLUX Restoration on Server (GPU {device_id})...")

    # 2. 加载 FLUX.1-dev (全精度 BF16/FP16)
    # 服务器显存够大，可以直接加载 dev 版本
    pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev", 
        torch_dtype=torch.float16  # 如果是 A100/H100，建议改用 torch.bfloat16
    )
    
    # 🔥 服务器性能全开模式：
    # 如果显存 > 24GB (如 A100, A6000)，直接 to(device) 最快
    # 如果显存 = 24GB (如 3090/4090)，Flux-Dev 可能会爆，还是建议保留 enable_model_cpu_offload()
    try:
        pipe.to(device)
        print("⚡ Model loaded directly to GPU (High Performance Mode)")
    except RuntimeError:
        print("⚠️ VRAM not enough for full load, falling back to CPU Offload")
        pipe.enable_model_cpu_offload()

    scheduler = pipe.scheduler

    for exp_dict in exp_configs:
        exp_name = exp_dict.get("exp_name", "FLUX_Restoration")
        model_type = "FLUX"
        
        # 参数读取
        T_steps = exp_dict.get("T_steps", 28) # FLUX 默认 28
        n_avg = exp_dict.get("n_avg", 1)
        n_min = exp_dict.get("n_min", 6)      # 去噪建议 6-10
        n_max = exp_dict.get("n_max", 24)
        
        src_guidance_scale = exp_dict.get("src_guidance_scale", 1.5)
        tar_guidance_scale = exp_dict.get("tar_guidance_scale", 5.5)
        
        seed = exp_dict.get("seed", 42)

        # 随机种子
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
        dataset_yaml = exp_dict["dataset_yaml"]
        dataset_configs = load_yaml(dataset_yaml)

        for data_dict in dataset_configs:
            src_prompt = data_dict["source_prompt"]
            tar_prompts = data_dict["target_prompts"]
            target_codes = data_dict.get("target_codes", [])
            
            image_src_path = data_dict["input_img"]

            if not os.path.exists(image_src_path):
                print(f"❌ Error: Image not found: {image_src_path}")
                continue

            # Load & Preprocess
            image = Image.open(image_src_path).convert("RGB")
            
            # ================= 服务器版分辨率设置 =================
            # 解锁到 2048 (4MP)，既能保证细节，又在 Flux 的舒适区内
            MAX_SIZE = 2048 
            
            w, h = image.size
            if w > MAX_SIZE or h > MAX_SIZE:
                print(f"   ⚠️ Large Image ({w}x{h}), cropping center {MAX_SIZE}x{MAX_SIZE}...")
                left = (w - MAX_SIZE) // 2
                top = (h - MAX_SIZE) // 2
                image = image.crop((left, top, left + MAX_SIZE, top + MAX_SIZE))
            
            # 16倍数对齐
            w, h = image.size
            new_w = w - (w % 16)
            new_h = h - (h % 16)
            if new_w != w or new_h != h:
                image = image.crop((0, 0, new_w, new_h))
            # ====================================================
            
            # Prepare Latents
            # Flux 的 latents 准备稍微不同，需要 pack
            # 但 pipe.prepare_latents 是内部调用的，我们这里只需要预处理图片
            image_src = pipe.image_processor.preprocess(image)
            image_src = image_src.to(device).half() # or bfloat16
            
            # Encode (Flux VAE)
            with torch.no_grad():
                x0_src_denorm = pipe.vae.encode(image_src).latent_dist.mode()
            
            x0_src = (x0_src_denorm - pipe.vae.config.shift_factor) * pipe.vae.config.scaling_factor
            
            # SIDD 路径解析 (防覆盖)
            path_parts = image_src_path.replace("\\", "/").split("/")
            if len(path_parts) >= 2:
                src_prompt_txt = f"{path_parts[-2]}_{path_parts[-1].split('.')[0]}"
            else:
                src_prompt_txt = path_parts[-1].split('.')[0]
            
            for tar_num, tar_prompt in enumerate(tar_prompts):
                tar_prompt_txt = target_codes[tar_num] if tar_num < len(target_codes) else str(tar_num)

                print(f"Processing: {src_prompt_txt} -> {tar_prompt_txt} | Res: {new_w}x{new_h}")

                # 调用 FlowEditFLUX
                x0_tar = FlowEditFLUX(
                    pipe,
                    scheduler,
                    x0_src,
                    src_prompt,
                    tar_prompt,
                    negative_prompt="", 
                    T_steps=T_steps,
                    n_avg=n_avg,
                    src_guidance_scale=src_guidance_scale,
                    tar_guidance_scale=tar_guidance_scale,
                    n_min=n_min,
                    n_max=n_max
                )

                # Decode
                x0_tar_denorm = (x0_tar / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor
                with torch.no_grad():
                    image_tar = pipe.vae.decode(x0_tar_denorm, return_dict=False)[0]
                
                image_tar = pipe.image_processor.postprocess(image_tar)
                
                # Save
                save_dir = f"outputs/{exp_name}/FLUX/{src_prompt_txt}/{tar_prompt_txt}"
                os.makedirs(save_dir, exist_ok=True)
                
                filename = f"nmin{n_min}_src{src_guidance_scale}_tar{tar_guidance_scale}.png"
                save_path = f"{save_dir}/{filename}"
                
                image_tar[0].save(save_path)
                print(f"   ✅ Saved to: {save_path}")

    print("Done")