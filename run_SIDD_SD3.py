import torch
from diffusers import StableDiffusion3Pipeline
from PIL import Image
import argparse
import random 
import numpy as np
import yaml
import os
import math
from tqdm import tqdm
# 导入 SD3 工具
from FlowEdit_utils import FlowEditSD3

def load_yaml(path):
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.load(f, Loader=yaml.FullLoader)

def process_tile_sd3(pipe, scheduler, image_tile, prompt_config, run_args, device):
    """
    处理单个 1024x1024 的小块 (SD3 版本)
    """
    # 1. 预处理
    # SD3 需要宽和高是 16 的倍数 (虽然 1024 肯定是，但加个保险)
    w, h = image_tile.size
    w = w - (w % 16)
    h = h - (h % 16)
    if w != image_tile.size[0] or h != image_tile.size[1]:
        image_tile = image_tile.crop((0, 0, w, h))

    image_src = pipe.image_processor.preprocess(image_tile)
    image_src = image_src.to(device).half()
    
    # 2. VAE 编码
    with torch.autocast("cuda"), torch.inference_mode():
        x0_src_denorm = pipe.vae.encode(image_src).latent_dist.mode()
    
    x0_src = (x0_src_denorm - pipe.vae.config.shift_factor) * pipe.vae.config.scaling_factor

    # 3. FlowEdit SD3 修复
    # 注意：这里不再传递 coupling_strength
    x0_tar = FlowEditSD3(
        pipe,
        scheduler,
        x0_src,
        src_prompt=prompt_config["source"],
        tar_prompt=prompt_config["target"],
        negative_prompt="", # SD3 默认空负面提示词
        T_steps=run_args["steps"],
        n_avg=1,
        src_guidance_scale=run_args["src_cfg"],
        tar_guidance_scale=run_args["tar_cfg"],
        n_min=run_args["n_min"],
        n_max=run_args["n_max"]
    )

    # 4. VAE 解码
    x0_tar_denorm = (x0_tar / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor
    with torch.autocast("cuda"), torch.inference_mode():
        image_tar = pipe.vae.decode(x0_tar_denorm, return_dict=False)[0]
    
    # 转回 Tensor (C, H, W) 范围 [0, 1]，方便拼图
    # image_tar 是 [-1, 1]，需要 denormalize
    image_tar = (image_tar / 2 + 0.5).clamp(0, 1).cpu().squeeze(0)
    return image_tar

def tiled_inference_sd3(pipe, scheduler, image_path, prompt_config, args, device):
    """
    核心分块逻辑：滑窗处理 + 加权融合
    """
    full_image = Image.open(image_path).convert("RGB")
    W, H = full_image.size
    
    # === 分块配置 ===
    # SD3 最佳分辨率是 1024
    TILE_SIZE = 1024    
    # 步长 768，意味着有 256px 的重叠区域用于平滑接缝
    STRIDE = 768        
    
    # 创建大画布 (累加器)
    full_canvas = torch.zeros((3, H, W), dtype=torch.float32)
    count_canvas = torch.zeros((3, H, W), dtype=torch.float32)

    print(f"   🧩 Tiling: {W}x{H} -> Grid with {TILE_SIZE}x{TILE_SIZE} tiles...")

    # 生成滑窗坐标
    h_starts = list(range(0, H - TILE_SIZE + 1, STRIDE))
    if (H - TILE_SIZE) % STRIDE != 0: h_starts.append(H - TILE_SIZE)
    
    w_starts = list(range(0, W - TILE_SIZE + 1, STRIDE))
    if (W - TILE_SIZE) % STRIDE != 0: w_starts.append(W - TILE_SIZE)
    
    h_starts = sorted(list(set(h_starts)))
    w_starts = sorted(list(set(w_starts)))

    total_tiles = len(h_starts) * len(w_starts)
    pbar = tqdm(total=total_tiles, desc="Processing Tiles", leave=False)

    for y in h_starts:
        for x in w_starts:
            # 1. 切片
            box = (x, y, x + TILE_SIZE, y + TILE_SIZE)
            tile_pil = full_image.crop(box)
            
            # 2. 处理 (FlowEdit SD3)
            tile_tensor = process_tile_sd3(
                pipe, scheduler, tile_pil, 
                prompt_config, 
                args, 
                device
            )
            
            # 3. 拼回去
            full_canvas[:, y:y+TILE_SIZE, x:x+TILE_SIZE] += tile_tensor
            count_canvas[:, y:y+TILE_SIZE, x:x+TILE_SIZE] += 1.0
            
            pbar.update(1)
    
    pbar.close()

    # 4. 取平均
    result_tensor = full_canvas / count_canvas
    
    # 转回 PIL
    result_img = result_tensor.permute(1, 2, 0).numpy() # (H, W, 3)
    result_img = (result_img * 255).astype(np.uint8)
    return Image.fromarray(result_img)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device_number", type=int, default=0, help="GPU ID")
    parser.add_argument("--exp_yaml", type=str, default="sidd_denoise.yaml")
    args_cli = parser.parse_args()
    
    device = torch.device(f"cuda:{args_cli.device_number}" if torch.cuda.is_available() else "cpu")
    exp_configs = load_yaml(args_cli.exp_yaml)

    print(f"🚀 Initializing SD3 Tiled Restoration (Server Mode)...")

    # 加载 SD3
    pipe = StableDiffusion3Pipeline.from_pretrained(
        "stabilityai/stable-diffusion-3-medium-diffusers", 
        torch_dtype=torch.float16
    )
    
    # 服务器模式：直接上 GPU
    try:
        pipe.to(device)
        print("⚡ Model loaded directly to GPU")
    except:
        print("⚠️ Falling back to CPU Offload")
        pipe.enable_model_cpu_offload()
    
    scheduler = pipe.scheduler

    for exp_dict in exp_configs:
        exp_name = exp_dict.get("exp_name", "SD3_Tiled")
        
        # 提取参数
        run_args = {
            "steps": exp_dict.get("T_steps", 50),
            "n_min": exp_dict.get("n_min", 20),
            "n_max": exp_dict.get("n_max", 45),
            "src_cfg": exp_dict.get("src_guidance_scale", 4.5),
            "tar_cfg": exp_dict.get("tar_guidance_scale", 9.0)
        }
        
        seed = exp_dict.get("seed", 42)
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

        dataset_configs = load_yaml(exp_dict["dataset_yaml"])

        for data_dict in dataset_configs:
            image_src_path = data_dict["input_img"]
            
            # ================= 关键修改：筛选逻辑 =================
            # 这里的逻辑是：检查路径字符串中是否包含 "_N" 
            # (SIDD 命名习惯: ..._3200_N, ..._4400_L)
            # 如果不包含，就跳过
            if "_N" not in image_src_path and "_N/" not in image_src_path:
                # print(f"Skipping non-'N' image: {image_src_path}")
                continue
            # ====================================================

            if not os.path.exists(image_src_path): continue

            # 准备 Prompt
            prompt_config = {
                "source": data_dict["source_prompt"],
                "target": data_dict["target_prompts"][0]
            }
            
            # 获取 Scene ID
            path_parts = image_src_path.replace("\\", "/").split("/")
            if len(path_parts) >= 2:
                # 例如: 0002_001_S6_..._N
                scene_id = path_parts[-2]
            else:
                scene_id = path_parts[-1].split('.')[0]
            
            print(f"🖼️ Processing Tiled SD3: {scene_id} ...")
            
            # 调用 Tiling 函数
            final_image = tiled_inference_sd3(
                pipe, scheduler, image_src_path, 
                prompt_config, run_args, device
            )
            
            # 保存
            # 结构: outputs/实验名/SD3_Tiled/SceneID/参数.png
            save_dir = f"outputs/{exp_name}/SD3_Tiled/{scene_id}"
            os.makedirs(save_dir, exist_ok=True)
            
            filename = f"nmin{run_args['n_min']}_src{run_args['src_cfg']}_tar{run_args['tar_cfg']}.png"
            save_path = f"{save_dir}/{filename}"
            
            final_image.save(save_path)
            print(f"✅ Saved: {save_path}")

    print("Done! All 'N' images processed.")