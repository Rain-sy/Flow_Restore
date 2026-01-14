import torch
from diffusers import StableDiffusion3Pipeline
# from diffusers import FluxPipeline # 暂时只用 SD3，Flux 先注释掉
from PIL import Image
import argparse
import random 
import numpy as np
import yaml
import os

# 注意：这里从 FlowRestore_utils 导入你新写的 FlowRestoreSD3
from FlowRestore_utils import FlowRestoreSD3

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--device_number", type=int, default=0, help="device number to use")
    # 默认加载 restore 的 yaml
    parser.add_argument("--exp_yaml", type=str, default="SD3_restore.yaml", help="experiment yaml file")

    args = parser.parse_args()

    # set device
    device_number = args.device_number
    device = torch.device(f"cuda:{device_number}" if torch.cuda.is_available() else "cpu")

    # load exp yaml file to dict
    exp_yaml = args.exp_yaml
    with open(exp_yaml, encoding='utf-8') as file:
        exp_configs = yaml.load(file, Loader=yaml.FullLoader)

    model_type = exp_configs[0]["model_type"] 

    if model_type == 'FLUX':
        # pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.float16)
        raise NotImplementedError("FlowRestore for FLUX is not yet integrated in this script.")
    elif model_type == 'SD3':
        # 加载 SD3
        print(f"Loading SD3 model on device {device_number}...")
        pipe = StableDiffusion3Pipeline.from_pretrained(
            "stabilityai/stable-diffusion-3-medium-diffusers", 
            torch_dtype=torch.float16,
            text_encoder_3=None, # 如果显存紧张，可以不加载 T5
            tokenizer_3=None
        )
    else:
        raise NotImplementedError(f"Model type {model_type} not implemented")
              
    scheduler = pipe.scheduler
    pipe = pipe.to(device)
    # pipe.enable_model_cpu_offload()   # 如果显存够大(24G+)，建议注释掉这就话，直接to(device)速度更快

    for exp_dict in exp_configs:

        exp_name = exp_dict["exp_name"]
        T_steps = exp_dict["T_steps"]
        n_avg = exp_dict["n_avg"]
        
        # --- 参数映射 ---
        # 原本的 src_guidance -> degradation_guidance (去噪力度)
        degradation_guidance_scale = exp_dict["degradation_guidance_scale"] 
        # 原本的 tar_guidance -> clean_guidance (清晰引导力度)
        clean_guidance_scale = exp_dict["clean_guidance_scale"]
        
        n_min = exp_dict["n_min"]
        n_max = exp_dict["n_max"]
        
        # --- 新增参数: Regularization ---
        reg_scale = exp_dict.get("reg_scale", 0.0) # 如果yaml里没写，默认为0
        
        seed = exp_dict["seed"]

        # set seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
        dataset_yaml = exp_dict["dataset_yaml"]
        with open(dataset_yaml, encoding='utf-8') as file:
            dataset_configs = yaml.load(file, Loader=yaml.FullLoader)

        for data_dict in dataset_configs:

            # 在 Restoration 任务中：
            # source_prompt -> 描述当前的坏图 (Degradation Prompt)
            degradation_prompt = data_dict["source_prompt"] 
            
            # target_prompts -> 描述目标 (Clean Prompt), 通常是一个列表，哪怕只有一个空字符串
            clean_prompts = data_dict["target_prompts"]
            
            negative_prompt = "" 
            image_src_path = data_dict["input_img"]

            print(f"Processing: {image_src_path}")

            # load image
            image = Image.open(image_src_path).convert("RGB")
            # crop image to have both dimensions divisibe by 16
            image = image.crop((0, 0, image.width - image.width % 16, image.height - image.height % 16))
            
            # 预处理
            image_src = pipe.image_processor.preprocess(image)
            image_src = image_src.to(device).half()
            
            with torch.autocast("cuda"), torch.inference_mode():
                # Encode 到 Latent
                x0_src_denorm = pipe.vae.encode(image_src).latent_dist.mode()
            
            # SD3 的 VAE Scaling
            x0_src = (x0_src_denorm - pipe.vae.config.shift_factor) * pipe.vae.config.scaling_factor
            x0_src = x0_src.to(device)
            
            # 这里的 x0_src 其实就是我们的 x_lq (Low Quality Input)
            
            for tar_num, clean_prompt in enumerate(clean_prompts):

                if model_type == 'SD3':
                    # 调用 FlowRestoreSD3
                    x0_tar = FlowRestoreSD3(
                        pipe=pipe,
                        scheduler=scheduler,
                        x_lq=x0_src,                        # 输入坏图 Latent
                        degradation_prompt=degradation_prompt, # 描述坏图
                        clean_prompt=clean_prompt,          # 描述好图 (或空)
                        negative_prompt=negative_prompt,
                        T_steps=T_steps,
                        n_avg=n_avg,
                        degradation_guidance_scale=degradation_guidance_scale,
                        clean_guidance_scale=clean_guidance_scale,
                        n_min=n_min,
                        n_max=n_max,
                        reg_scale=reg_scale,                # 正则化参数
                        degradation_fn=None                 # 默认为 None (Denoise)
                    )
                                
                else:
                    raise NotImplementedError(f"Model type {model_type} not implemented")

                # Decode 回像素空间
                x0_tar_denorm = (x0_tar / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor
                
                with torch.autocast("cuda"), torch.inference_mode():
                    image_tar = pipe.vae.decode(x0_tar_denorm, return_dict=False)[0]
                
                image_tar = pipe.image_processor.postprocess(image_tar)

                # --- 保存路径逻辑 (保持原样) ---
                # 获取文件名 (不带后缀)
                src_filename = image_src_path.replace("\\", "/").split("/")[-1].split(".")[0]
                
                tar_idx_str = str(tar_num)
                
                # 路径: outputs/实验名/SD3/图片名/target序号/
                save_dir = f"outputs/{exp_name}/{model_type}/src_{src_filename}/tar_{tar_idx_str}"
                os.makedirs(save_dir, exist_ok=True)
                
                # 文件名包含关键参数，增加 reg_scale
                output_filename = (
                    f"n_min_{n_min}_n_max_{n_max}_"
                    f"deg{degradation_guidance_scale}_clean{clean_guidance_scale}_"
                    f"reg{reg_scale}_steps{T_steps}.png"
                )
                
                save_path = f"{save_dir}/{output_filename}"
                image_tar[0].save(save_path)
                
                print(f"Saved to {save_path}")

    print("All restoration tasks done.")