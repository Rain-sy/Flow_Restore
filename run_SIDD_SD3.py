import argparse
import yaml
import torch
import os
import numpy as np
from PIL import Image
from diffusers import StableDiffusion3Pipeline
# 确保你的 FlowEdit_utils 已经修改过，支持 coupling_strength
from FlowEdit_utils import FlowEditSD3

def load_yaml(path):
    """安全加载包含中文路径的 YAML"""
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.load(f, Loader=yaml.FullLoader)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device_number", type=int, default=0, help="GPU device ID")
    parser.add_argument("--exp_yaml", type=str, default="SD3_denoise.yaml", help="配置文件路径")
    args = parser.parse_args()
    
    device_id = args.device_number
    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
    
    pipe = StableDiffusion3Pipeline.from_pretrained(
        "stabilityai/stable-diffusion-3-medium-diffusers", 
        torch_dtype=torch.float16
    )
    
    pipe = pipe.to(device)
    
    # 获取 Scheduler
    scheduler = pipe.scheduler

    # 3. 读取实验配置
    exp_configs = load_yaml(args.exp_yaml)

    # 4. 开始循环实验
    for exp_dict in exp_configs:
        exp_name = exp_dict.get("exp_name", "SD3_Restoration")
        dataset_yaml = exp_dict["dataset_yaml"]
        
        # 读取参数 (提供默认值以防 YAML 漏写)
        T_steps = exp_dict.get("T_steps", 50)
        n_avg = exp_dict.get("n_avg", 1)
        n_min = exp_dict.get("n_min", 15)
        n_max = exp_dict.get("n_max", 45)
        
        # 关键参数
        src_guidance_scale = exp_dict.get("src_guidance_scale", 4.5)
        tar_guidance_scale = exp_dict.get("tar_guidance_scale", 8.5)
        coupling_strength = exp_dict.get("coupling_strength", 0.6) # 默认 0.6
        
        seed = exp_dict.get("seed", 42)

        print(f"\n🎨 Starting Experiment: {exp_name}")
        print(f"   Steps: {T_steps} | Coupling: {coupling_strength} | n_min: {n_min}")
        print(f"   Guidance: Src={src_guidance_scale} / Tar={tar_guidance_scale}")

        # 设置随机种子
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)

        # 加载数据配置
        dataset_configs = load_yaml(dataset_yaml)

        for data_dict in dataset_configs:
            image_path = data_dict["input_img"]
            src_prompt = data_dict["source_prompt"]
            tar_prompts = data_dict["target_prompts"]
            
            # 获取 target_codes (用于命名)，如果没有就用索引
            target_codes = data_dict.get("target_codes", [])

            # --- 图像预处理 ---
            if not os.path.exists(image_path):
                print(f"❌ Error: Image not found: {image_path}")
                continue
            
            # 强制转 RGB (修复 4 通道报错)
            image_raw = Image.open(image_path)
            
            # 调整尺寸为 16 的倍数 (避免 VAE 报错)
            w, h = image_raw.size
            new_w = w - (w % 16)
            new_h = h - (h % 16)
            if new_w != w or new_h != h:
                image_raw = image_raw.resize((new_w, new_h), Image.LANCZOS)

            # 预处理并编码进 VAE
            image_tensor = pipe.image_processor.preprocess(image_raw).to(device).half()
            
            with torch.no_grad():
                # 编码 Latents
                x0_src_denorm = pipe.vae.encode(image_tensor).latent_dist.mode()
                # 归一化 Latents
                x0_src = (x0_src_denorm - pipe.vae.config.shift_factor) * pipe.vae.config.scaling_factor

            # --- 开始编辑 ---
            for i, tar_prompt in enumerate(tar_prompts):
                
                # 确定输出标识
                code_suffix = target_codes[i] if i < len(target_codes) else str(i)
                
                print(f"   Processing: {os.path.basename(image_path)} -> {code_suffix}")

                # 调用核心算法 (确保 utils 里有 coupling_strength 参数)
                x0_tar = FlowEditSD3(
                    pipe,
                    scheduler,
                    x0_src,
                    src_prompt,
                    tar_prompt,
                    negative_prompt="", # SD3 通常不需要显式负面提示词，除非 guidance > 1
                    T_steps=T_steps,
                    n_avg=n_avg,
                    src_guidance_scale=src_guidance_scale,
                    tar_guidance_scale=tar_guidance_scale,
                    n_min=n_min,
                    n_max=n_max,
                    coupling_strength=coupling_strength # <--- 传入这个关键参数
                )

                # 解码 Latents -> Image
                x0_tar_denorm = (x0_tar / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor
                with torch.no_grad():
                    image_out = pipe.vae.decode(x0_tar_denorm, return_dict=False)[0]
                
                image_out = pipe.image_processor.postprocess(image_out)[0]

                # --- 智能保存逻辑 ---
                # 1. 解析路径结构，提取父文件夹名 (适配 SIDD 等复杂数据集)
                # 将路径统一为正斜杠处理
                path_parts = image_path.replace("\\", "/").split("/")
                
                if len(path_parts) >= 2:
                    # 例如 Data/001/NOISY.png -> 001_NOISY
                    src_folder_name = f"{path_parts[-2]}_{path_parts[-1].split('.')[0]}"
                else:
                    src_folder_name = path_parts[-1].split('.')[0]

                # 2. 构建保存目录
                # output/实验名/SD3/原图ID/目标ID
                save_dir = f"outputs/{exp_name}/SD3/{src_folder_name}/{code_suffix}"
                os.makedirs(save_dir, exist_ok=True)

                # 3. 构建详细文件名 (带参数)
                filename = (f"cp{coupling_strength}_nmin{n_min}_"
                            f"src{src_guidance_scale}_tar{tar_guidance_scale}.png")
                
                save_path = os.path.join(save_dir, filename)
                image_out.save(save_path)
                print(f"      ✅ Saved: {save_path}")

    print("\n🎉 All Done!")