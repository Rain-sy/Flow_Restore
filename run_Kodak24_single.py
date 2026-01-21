import torch
import os
from PIL import Image
from diffusers import FluxPipeline
import numpy as np

# 确保 FlowRestore_utils.py 在同一目录下
from FlowRestore_utils import FlowRestoreFLUX

def load_image(image_path, pipe):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
        
    image = Image.open(image_path).convert("RGB")
    w, h = image.size
    
    # 裁剪为 16 的倍数 (适配 FLUX)
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
    output_path = "outputs/Single_Test/kodim04_sigma50_detailed.png"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 3. 详细的 Source Prompt (描述图片内容 + 噪声描述)
    # 注意：我们在描述内容的后面，加上噪声的描述
    content_desc = (
        "a woman with blonde hair, wearing a large red woven hat and a scarf. "
    )
    # 组合 Prompt
    degradation_prompt = f"A noised image of {content_desc}"

    # 4. Target Prompt 设为空 (实现 Prompt Subtraction)
    clean_prompt = f"A denoised image of {content_desc}"
    
    # 5. 参数设置 (针对 Sigma 50)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    T_steps = 30
    # Sigma 50 噪声很大，Guidance Scale 需要给劲一点
    degradation_guidance_scale = 2.0 
    clean_guidance_scale = 2.0  # 空 Prompt 时建议设为 1.0 或 1.5
    tolerance = 1e-4            # 自适应停止阈值 (更严格)
    min_steps = 10             # 最小步数
    # ===========================================

    print(f"Loading FLUX model on {device}...")
    pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev", 
        torch_dtype=torch.float16
    ).to(device)

    print(f"Processing {img_path}...")
    
    # 1. 加载并编码
    input_tensor = load_image(img_path, pipe).to(device).to(torch.float16)
    with torch.no_grad():
        x_lq = pipe.vae.encode(input_tensor).latent_dist.mode()
        x_lq = (x_lq - pipe.vae.config.shift_factor) * pipe.vae.config.scaling_factor

    # 2. 运行 FlowRestore
    print(f"Source Prompt: {degradation_prompt}")
    print(f"Target Prompt: {clean_prompt}")
    
    restored_latents = FlowRestoreFLUX(
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
        n_max=T_steps-8,
        reg_scale=0.0,  # 去噪不开正则化
        adaptive_tolerance=tolerance,
        min_inference_steps=min_steps
    )

    # 3. 解码并保存
    print("Decoding and saving...")
    with torch.no_grad():
        restored_latents = (restored_latents / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor
        image_out = pipe.vae.decode(restored_latents, return_dict=False)[0]
        image_out = pipe.image_processor.postprocess(image_out, output_type="pil")[0]

    image_out.save(output_path)
    print(f"Done! Saved to {output_path}")

if __name__ == "__main__":
    main()