import torch
import os
from PIL import Image
from diffusers import StableDiffusion3Pipeline
import numpy as np

# 确保 FlowRestore_utils.py 在同一目录下
# 并且其中包含 FlowRestoreSD3 函数
from FlowRestore_utils import FlowRestoreSD3

def load_image(image_path, pipe):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
        
    image = Image.open(image_path).convert("RGB")
    w, h = image.size
    
    # 裁剪为 16 的倍数 (SD3 VAE 同样建议对齐)
    new_w = w - (w % 16)
    new_h = h - (h % 16)
    if new_w != w or new_h != h:
        image = image.crop((0, 0, new_w, new_h))
    
    # 预处理
    image = pipe.image_processor.preprocess(image)
    return image

def main():
    # ================= 配置区域 =================
    # 1. 指定那张特定的 Sigma 50 图片
    img_path = "Data/Kodak24_noised/sigma_50/kodim04.png"
    
    # 2. 输出路径
    output_path = "outputs/Single_Test/kodim04_sigma50_sd3_paired.png"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 3. 详细的内容描述
    content_desc = "a woman with blonde hair, wearing a red hat and a scarf"
    
    # 策略：Paired Prompts (成对提示词)
    # Source: 描述噪声状态
    degradation_prompt = f"A noised image of {content_desc}"
    
    # Target: 描述清晰状态
    # SD3 对 "High quality" 等词汇响应很好
    clean_prompt = f"A denoised image of {content_desc}"
    
    # 5. 参数设置 (针对 Sigma 50 + SD3)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    T_steps = 30
    
    # SD3 参数建议：
    # deg_scale: 4.0 ~ 5.0 (推离噪声)
    degradation_guidance_scale = 3.5 
    # clean_scale: 1.5 ~ 2.0 (拉向清晰，SD3 即使给 1.0 以上通常也不会像 FLUX 那样容易脸崩，可以稍微给高一点试试)
    clean_guidance_scale = 3.5       
    
    tolerance = 1e-5             # 自适应停止阈值 (SD3 轨迹比 FLUX 稍弯曲一点，1e-5 比较安全)
    min_steps = 35               # 最小步数
    # ===========================================

    print(f"Loading SD3 model on {device}...")
    pipe = StableDiffusion3Pipeline.from_pretrained(
        "stabilityai/stable-diffusion-3-medium-diffusers", 
        torch_dtype=torch.float16,
        text_encoder_3=None, # 去噪任务可以不加载 T5 以节省显存
        tokenizer_3=None
    ).to(device)

    print(f"Processing {img_path}...")
    
    # 1. 加载并编码
    input_tensor = load_image(img_path, pipe).to(device).to(torch.float16)
    
    with torch.no_grad():
        # VAE Encode
        x_lq = pipe.vae.encode(input_tensor).latent_dist.mode()
        # SD3 的 Scaling 逻辑
        x_lq = (x_lq - pipe.vae.config.shift_factor) * pipe.vae.config.scaling_factor

    # 2. 运行 FlowRestoreSD3
    print(f"Source Prompt: {degradation_prompt}")
    print(f"Target Prompt: {clean_prompt}")
    print(f"Guidance: Deg={degradation_guidance_scale}, Clean={clean_guidance_scale}")
    
    restored_latents = FlowRestoreSD3(
        pipe=pipe,
        scheduler=pipe.scheduler,
        x_lq=x_lq,
        degradation_prompt=degradation_prompt,
        clean_prompt=clean_prompt,
        T_steps=T_steps,
        n_avg=1,
        degradation_guidance_scale=degradation_guidance_scale,
        clean_guidance_scale=clean_guidance_scale,
        n_min=0,
        n_max=T_steps-7,
        reg_scale=0.0,  # 去噪不开正则化
        adaptive_tolerance=tolerance,
        min_inference_steps=min_steps
    )

    # 3. 解码并保存
    print("Decoding and saving...")
    with torch.no_grad():
        # Unscale
        restored_latents = (restored_latents / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor
        image_out = pipe.vae.decode(restored_latents, return_dict=False)[0]
        image_out = pipe.image_processor.postprocess(image_out, output_type="pil")[0]

    image_out.save(output_path)
    print(f"Done! Saved to {output_path}")

if __name__ == "__main__":
    main()