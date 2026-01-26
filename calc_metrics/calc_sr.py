import os
import glob
import numpy as np
import torch
import lpips
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as psnr_loss
from skimage.metrics import structural_similarity as ssim_loss
from tqdm import tqdm

def find_gt_file(sr_filename, gt_files_set):
    """
    针对 Urban100 数据集的精确匹配逻辑
    SR: img_001_SRF_2_LR.png
    GT: img_001_SRF_2_HR.png
    """
    # 核心逻辑：直接将 _LR 替换为 _HR
    if "_LR" in sr_filename:
        target_gt = sr_filename.replace("_LR", "_HR")
        if target_gt in gt_files_set:
            return target_gt
    
    # 备用逻辑：万一有的文件没有 _LR 后缀，尝试直接匹配
    if sr_filename in gt_files_set:
        return sr_filename

    return None

def calculate_metrics(sr_dir, gt_dir, device='cuda'):
    print(f"Loading LPIPS model on {device}...")
    loss_fn_alex = lpips.LPIPS(net='alex').to(device)
    
    # 获取文件列表
    sr_files = sorted(glob.glob(os.path.join(sr_dir, "*.png")))
    gt_files = sorted(glob.glob(os.path.join(gt_dir, "*.png")))
    
    if not sr_files:
        print(f"[Error] No SR images found in: {sr_dir}")
        return
    if not gt_files:
        print(f"[Error] No GT images found in: {gt_dir}")
        return

    # 建立 GT 文件名集合 (用于快速查找)
    gt_files_set = set([os.path.basename(f) for f in gt_files])
    # 建立 GT 全路径字典
    gt_full_path_map = {os.path.basename(f): f for f in gt_files}

    metrics = {"psnr": [], "ssim": [], "lpips": []}
    matched_count = 0
    
    print(f"\nComparing SR ({len(sr_files)}) vs GT ({len(gt_files)})...")
    
    # 打印前几个文件名，确保我们看对了
    print(f"SR Example: {os.path.basename(sr_files[0])}")
    print(f"GT Example: {os.path.basename(gt_files[0])}")
    print("-" * 30)

    for sr_path in tqdm(sr_files):
        sr_filename = os.path.basename(sr_path)
        
        # 查找匹配的 GT 文件名
        gt_filename = find_gt_file(sr_filename, gt_files_set)
        
        if gt_filename:
            gt_path = gt_full_path_map[gt_filename]
            matched_count += 1
            
            # --- 读取与计算 ---
            img_sr = Image.open(sr_path).convert("RGB")
            img_gt = Image.open(gt_path).convert("RGB")
            
            # 尺寸对齐 (Top-Left Crop)
            w_min = min(img_sr.width, img_gt.width)
            h_min = min(img_sr.height, img_gt.height)
            
            img_sr = img_sr.crop((0, 0, w_min, h_min))
            img_gt = img_gt.crop((0, 0, w_min, h_min))
            
            sr_np = np.array(img_sr)
            gt_np = np.array(img_gt)
            
            # PSNR
            cur_psnr = psnr_loss(gt_np, sr_np, data_range=255)
            
            # SSIM
            cur_ssim = ssim_loss(gt_np, sr_np, data_range=255, channel_axis=2, win_size=11)
            
            # LPIPS
            sr_tensor = (torch.tensor(sr_np).permute(2, 0, 1).float() / 127.5) - 1.0
            gt_tensor = (torch.tensor(gt_np).permute(2, 0, 1).float() / 127.5) - 1.0
            sr_tensor = sr_tensor.unsqueeze(0).to(device)
            gt_tensor = gt_tensor.unsqueeze(0).to(device)
            
            with torch.no_grad():
                cur_lpips = loss_fn_alex(sr_tensor, gt_tensor).item()
                
            metrics["psnr"].append(cur_psnr)
            metrics["ssim"].append(cur_ssim)
            metrics["lpips"].append(cur_lpips)
            
        else:
            if matched_count < 3:
                print(f"[Warning] GT match failed for: {sr_filename} (Expected ..._HR.png)")

    if matched_count == 0:
        print("\n[CRITICAL ERROR] Matched 0 pairs!")
        return

    print("\n" + "="*40)
    print("Evaluation Results")
    print("="*40)
    print(f"Matched: {matched_count} images")
    print(f"Avg PSNR : {np.mean(metrics['psnr']):.4f}")
    print(f"Avg SSIM : {np.mean(metrics['ssim']):.4f}")
    print(f"Avg LPIPS: {np.mean(metrics['lpips']):.4f}")
    print("="*40)

if __name__ == "__main__":
    sr_dir_input = r"outputs/Urban100sr/FLUX"
    gt_dir_input = r"Data/Urban 100/X2 Urban100/X2/HIGH X2 Urban"
    
    calculate_metrics(sr_dir_input, gt_dir_input)