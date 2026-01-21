import os
import torch
import argparse
from PIL import Image
from diffusers import FluxPipeline, StableDiffusion3Pipeline
from tqdm import tqdm

# 导入你的 Utils
from FlowRestore_utils import FlowRestoreFLUX, FlowRestoreSD3

def parse_args():
    parser = argparse.ArgumentParser(description="Run FlowRestore on Kodak24")
    
    # 模型选择
    parser.add_argument("--model_type", type=str, default="FLUX", choices=["FLUX", "SD3"], help="Choose model: FLUX or SD3")
    
    # 路径
    parser.add_argument("--input_dir", type=str, default="Data/Kodak24_noised", help="Path to noised dataset")
    parser.add_argument("--output_dir", type=str, default="outputs/Kodak24_restored", help="Path to save results")
    
    # 自适应早停参数
    parser.add_argument("--tolerance", type=float, default=1e-3, help="Tolerance for adaptive stopping (FLUX: 1e-3, SD3: 1e-4)")
    parser.add_argument("--min_steps", type=int, default=10, help="Minimum steps before checking convergence")
    
    # 最大步数 (设大一点，让它自己停)
    parser.add_argument("--max_steps", type=int, default=30, help="Maximum inference steps")
    
    return parser.parse_args()

def load_image(image_path, pipe):
    """加载并预处理图片"""
    image = Image.open(image_path).convert("RGB")
    w, h = image.size
    
    # 裁剪为 16 的倍数 (适配 VAE)
    new_w = w - (w % 16)
    new_h = h - (h % 16)
    if new_w != w or new_h != h:
        image = image.crop((0, 0, new_w, new_h))
    
    # 预处理 [0,1] -> [-1,1] 并转为 Tensor
    image = pipe.image_processor.preprocess(image)
    return image

def main():
    args = parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on {device} with model {args.model_type}...")
    
    # --- 1. 加载模型 ---
    if args.model_type == "FLUX":
        model_id = "black-forest-labs/FLUX.1-dev"
        pipe = FluxPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to(device)
        # FLUX 推荐 Tolerance
        # if args.tolerance == 0: args.tolerance = 1e-3
        
    elif args.model_type == "SD3":
        model_id = "stabilityai/stable-diffusion-3-medium-diffusers"
        # 不加载 T5 (text_encoder_3) 以节省显存，Restoration 任务通常不需要 T5 的复杂语义
        pipe = StableDiffusion3Pipeline.from_pretrained(
            model_id, 
            torch_dtype=torch.float16,
            text_encoder_3=None, 
            tokenizer_3=None
        ).to(device)
        # SD3 推荐 Tolerance
        # if args.tolerance == 0: args.tolerance = 1e-4

    # --- 2. 遍历噪声等级 ---
    sigmas = [10]
    
    for sigma in sigmas:
        curr_input_dir = os.path.join(args.input_dir, f"sigma_{sigma}")
        curr_output_dir = os.path.join(args.output_dir, args.model_type, f"sigma_{sigma}") # 分开保存
        os.makedirs(curr_output_dir, exist_ok=True)
        
        if not os.path.exists(curr_input_dir):
            print(f"Skipping sigma {sigma}: {curr_input_dir} not found.")
            continue
            
        print(f"\nProcessing Sigma {sigma} with Adaptive Stopping (Tol={args.tolerance})...")
        
        # --- 3. 动态配置参数 ---
        if sigma <= 15:
            deg_scale = 2.0 if args.model_type == "FLUX" else 4.0
            # clean_scale = 2.0 if args.model_type == "FLUX" else 4.0
        elif sigma <= 30:
            deg_scale = 3.0 if args.model_type == "FLUX" else 5.0
            # clean_scale = 3.0 if args.model_type == "FLUX" else 5.0
        else:
            deg_scale = 4.0 if args.model_type == "FLUX" else 6.0
            # clean_scale = 4.0 if args.model_type == "FLUX" else 6.0
        # deg_scale = 2.5
        clean_scale = 1.5 if args.model_type == "FLUX" else 1.0
        deg_prompt = "An image added with Gaussian noise"
        clean_prompt = "A denoised image"
        
        # 获取图片列表
        image_files = [f for f in os.listdir(curr_input_dir) if f.endswith(('.png', '.jpg'))]
        
        for img_name in tqdm(image_files, desc=f"Sigma {sigma}"):
            img_path = os.path.join(curr_input_dir, img_name)
            save_path = os.path.join(curr_output_dir, img_name)
            
            # 加载并编码
            input_tensor = load_image(img_path, pipe).to(device).to(torch.float16)
            
            with torch.no_grad():
                x_lq = pipe.vae.encode(input_tensor).latent_dist.mode()
                # 手动 Scaling
                x_lq = (x_lq - pipe.vae.config.shift_factor) * pipe.vae.config.scaling_factor
            
            # --- 4. 运行 Restore (根据模型分发) ---
            if args.model_type == "FLUX":
                restored_latents = FlowRestoreFLUX(
                    pipe=pipe,
                    scheduler=pipe.scheduler,
                    x_lq=x_lq,
                    degradation_prompt=deg_prompt,
                    clean_prompt=clean_prompt,
                    T_steps=args.max_steps,        # 给最大步数
                    degradation_guidance_scale=deg_scale,
                    clean_guidance_scale=clean_scale,
                    n_min=0,
                    n_max=args.max_steps,
                    reg_scale=0.0,                 # 去噪通常不开正则化
                    adaptive_tolerance=args.tolerance, # 自适应阈值
                    min_inference_steps=args.min_steps
                )
            else: # SD3
                restored_latents = FlowRestoreSD3(
                    pipe=pipe,
                    scheduler=pipe.scheduler,
                    x_lq=x_lq,
                    degradation_prompt=deg_prompt,
                    clean_prompt=clean_prompt,
                    T_steps=args.max_steps,
                    degradation_guidance_scale=deg_scale,
                    clean_guidance_scale=clean_scale, # 通常为 1.0
                    n_min=0,
                    n_max=args.max_steps,
                    reg_scale=0.0, # 如果想要 SD3 保持更好结构，可以给一点点 reg_scale=50.0
                    adaptive_tolerance=args.tolerance,
                    min_inference_steps=args.min_steps
                )
            
            # --- 5. 解码并保存 ---
            with torch.no_grad():
                # Unscale
                restored_latents = (restored_latents / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor
                image_out = pipe.vae.decode(restored_latents, return_dict=False)[0]
                image_out = pipe.image_processor.postprocess(image_out, output_type="pil")[0]
            
            image_out.save(save_path)

    print(f"\nProcessing complete. Results saved in {args.output_dir}")

if __name__ == "__main__":
    main()