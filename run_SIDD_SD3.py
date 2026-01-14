import torch
from diffusers import StableDiffusion3Pipeline
from PIL import Image
import argparse
import random 
import numpy as np
import yaml
import os
# 只需要导入 SD3 的工具
from FlowEdit_utils import FlowEditSD3

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--device_number", type=int, default=0, help="device number to use")
    parser.add_argument("--exp_yaml", type=str, default="sidd_denoise.yaml", help="experiment yaml file")

    args = parser.parse_args()

    # set device
    device_number = args.device_number
    # 这里定义 device 变量，但后面主要靠 cpu_offload 管理
    device = torch.device(f"cuda:{device_number}" if torch.cuda.is_available() else "cpu")

    # load exp yaml file to dict
    exp_yaml = args.exp_yaml
    with open(exp_yaml, encoding='utf-8') as file:
        exp_configs = yaml.load(file, Loader=yaml.FullLoader)

    print(f"🚀 Initializing SD3 for SIDD Restoration...")

    # 1. 加载 SD3 模型
    # 既然只用 SD3，直接写死加载逻辑，不再判断 model_type
    pipe = StableDiffusion3Pipeline.from_pretrained(
        "stabilityai/stable-diffusion-3-medium-diffusers", 
        torch_dtype=torch.float16
    )
    
    scheduler = pipe.scheduler
    
    # 2. 开启 CPU Offload (8GB 显存优化)
    print("💡 Enabling Model CPU Offload...")
    pipe.enable_model_cpu_offload()

    for exp_dict in exp_configs:

        exp_name = exp_dict["exp_name"]
        model_type = "SD3" # 固定为 SD3
        
        T_steps = exp_dict["T_steps"]
        n_avg = exp_dict["n_avg"]
        src_guidance_scale = exp_dict["src_guidance_scale"]
        tar_guidance_scale = exp_dict["tar_guidance_scale"]
        n_min = exp_dict["n_min"]
        n_max = exp_dict["n_max"]
        seed = exp_dict["seed"]

        # set seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
        dataset_yaml = exp_dict["dataset_yaml"]
        with open(dataset_yaml, encoding='utf-8') as file:
            dataset_configs = yaml.load(file, Loader=yaml.FullLoader)

        # 遍历数据集图片
        for data_dict in dataset_configs:

            src_prompt = data_dict["source_prompt"]
            tar_prompts = data_dict["target_prompts"]
            
            # 获取 target_codes (如果 YAML 里有就用，没有就用索引)
            target_codes = data_dict.get("target_codes", [])
            
            negative_prompt = "" 
            image_src_path = data_dict["input_img"]

            # check image existence
            if not os.path.exists(image_src_path):
                print(f"❌ Error: Image not found: {image_src_path}")
                continue

            # load image
            # 强制转 RGB，防止 PNG 的 Alpha 通道导致报错
            image = Image.open(image_src_path).convert("RGB")
            
            # crop image to have both dimensions divisibe by 16
            # 使用 LANCZOS 缩放通常比直接 crop 更好，但保留你的 crop 逻辑也行
            # 这里稍微优化了一下逻辑，确保 crop 不会出错
            w, h = image.size
            new_w = w - (w % 16)
            new_h = h - (h % 16)
            if new_w != w or new_h != h:
                image = image.crop((0, 0, new_w, new_h))
            
            image_src = pipe.image_processor.preprocess(image)
            
            # cast image to half precision
            image_src = image_src.to(device).half()
            
            # VAE Encode
            with torch.autocast("cuda"), torch.inference_mode():
                x0_src_denorm = pipe.vae.encode(image_src).latent_dist.mode()
            
            x0_src = (x0_src_denorm - pipe.vae.config.shift_factor) * pipe.vae.config.scaling_factor
            x0_src = x0_src.to(device)
            
            # ================= SIDD 路径处理核心逻辑 =================
            # SIDD 的图片名都是 NOISY_SRGB_010.PNG，如果不处理会覆盖
            # 这里的逻辑是：如果路径包含父文件夹，就把父文件夹名拼上去
            path_parts = image_src_path.replace("\\", "/").split("/")
            
            if len(path_parts) >= 2:
                # 例如: 001_NOISY_SRGB_010
                src_prompt_txt = f"{path_parts[-2]}_{path_parts[-1].split('.')[0]}"
            else:
                src_prompt_txt = path_parts[-1].split('.')[0]
            # =======================================================
            
            for tar_num, tar_prompt in enumerate(tar_prompts):

                # 确定目标文件夹名称
                if tar_num < len(target_codes):
                    tar_prompt_txt = target_codes[tar_num]
                else:
                    tar_prompt_txt = str(tar_num)

                print(f"Processing: {src_prompt_txt} -> {tar_prompt_txt}")

                # 调用 FlowEditSD3 (已移除 coupling_strength)
                x0_tar = FlowEditSD3(
                    pipe,
                    scheduler,
                    x0_src,
                    src_prompt,
                    tar_prompt,
                    negative_prompt,
                    T_steps=T_steps,
                    n_avg=n_avg,
                    src_guidance_scale=src_guidance_scale,
                    tar_guidance_scale=tar_guidance_scale,
                    n_min=n_min,
                    n_max=n_max
                    # 注意：这里不再传递 coupling_strength
                )

                # Decode
                x0_tar_denorm = (x0_tar / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor
                with torch.autocast("cuda"), torch.inference_mode():
                    image_tar = pipe.vae.decode(x0_tar_denorm, return_dict=False)[0]
                
                image_tar = pipe.image_processor.postprocess(image_tar)
                
                # 构造保存路径
                # 结构: outputs/实验名/SD3/src_场景ID/tar_code/文件名.png
                save_dir = f"outputs/{exp_name}/{model_type}/src_{src_prompt_txt}/tar_{tar_prompt_txt}"
                os.makedirs(save_dir, exist_ok=True)
                
                output_filename = f"n_min_{n_min}_n_max_{n_max}_src{src_guidance_scale}_tar{tar_guidance_scale}_T_steps_{T_steps}.png"
                save_path = f"{save_dir}/{output_filename}"
                
                image_tar[0].save(save_path)
                print(f"   Saved to: {save_path}")

    print("Done")