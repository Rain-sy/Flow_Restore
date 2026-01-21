import os
import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr_loss
from skimage.metrics import structural_similarity as ssim_loss

# ================= 配置区域 =================
# 1. 指向 Tiled 推理的结果目录
# 例如: outputs/SD3_Restoration/SD3_Tiled
RESTORED_ROOT = "outputs/SIDD_FLUX_Denoise/SD3_Tiled"

# 2. SIDD 数据集原始目录 (用于找 4K GT)
SIDD_DATA_ROOT = "Data/SIDD_Small_sRGB_Only/Data"
# ===========================================

def calculate_metrics(img_gt, img_restored):
    # 1. 尺寸安全检查
    # 虽然 Tiled 推理应该还原原尺寸，但为了代码健壮性，取两者的最小公共区域
    h_res, w_res = img_restored.shape[:2]
    h_gt, w_gt = img_gt.shape[:2]
    
    if h_res != h_gt or w_res != w_gt:
        print(f"   ⚠️ Size mismatch! GT: {w_gt}x{h_gt}, Res: {w_res}x{h_res}. Cropping to overlap.")
        min_h = min(h_res, h_gt)
        min_w = min(w_res, w_gt)
        img_gt = img_gt[:min_h, :min_w]
        img_restored = img_restored[:min_h, :min_w]

    # 2. 转为 Y 通道 (亮度) 计算 (学术界标准)
    # 将 BGR 转为 Y 通道 (公式: 65.481*R + 128.553*G + 24.966*B + 16)
    img_gt = img_gt.astype(np.float32) / 255.
    img_restored = img_restored.astype(np.float32) / 255.
    
    # 注意：OpenCV 读取顺序是 BGR (img[..., 0]是B, 1是G, 2是R)
    img_gt_y = 65.481 * img_gt[..., 2] + 128.553 * img_gt[..., 1] + 24.966 * img_gt[..., 0] + 16.0
    img_restored_y = 65.481 * img_restored[..., 2] + 128.553 * img_restored[..., 1] + 24.966 * img_restored[..., 0] + 16.0
    
    img_gt_y /= 255.0
    img_restored_y /= 255.0

    # 3. 计算 PSNR 和 SSIM
    psnr = psnr_loss(img_gt_y, img_restored_y, data_range=1.0)
    ssim = ssim_loss(img_gt_y, img_restored_y, data_range=1.0)
    return psnr, ssim

def main():
    total_psnr = 0
    total_ssim = 0
    count = 0
    
    print(f"Scanning Full-Resolution results in: {RESTORED_ROOT}")
    
    # 遍历结果文件夹
    for root, dirs, files in os.walk(RESTORED_ROOT):
        for file in files:
            if not file.lower().endswith(('.png', '.jpg')): continue
            
            # 解析 Scene ID
            # 你的 Tiled 脚本生成的结构通常是: .../SD3_Tiled/{SceneID}/xxxx.png
            # 所以父文件夹名就是 Scene ID (例如: 0004_001_S6_..._4400_N)
            scene_id = os.path.basename(root)
            
            # 简单的校验：Scene ID 通常以数字开头
            if not scene_id[0].isdigit():
                continue
            
            # 构造 GT 路径
            gt_path = os.path.join(SIDD_DATA_ROOT, scene_id, "GT_SRGB_010.PNG")
            
            if not os.path.exists(gt_path):
                # 兼容 011 后缀
                gt_path = os.path.join(SIDD_DATA_ROOT, scene_id, "GT_SRGB_011.PNG")
                if not os.path.exists(gt_path):
                    # print(f"GT not found for scene: {scene_id}")
                    continue
            
            file_path = os.path.join(root, file)
            
            # 读取图片
            img_restored = cv2.imread(file_path)
            img_gt = cv2.imread(gt_path)
            
            if img_gt is None or img_restored is None: continue
            
            # 计算
            psnr, ssim = calculate_metrics(img_gt, img_restored)
            
            print(f"[{scene_id}] PSNR: {psnr:.2f} | SSIM: {ssim:.4f} | {file}")
            
            total_psnr += psnr
            total_ssim += ssim
            count += 1

    if count > 0:
        print("-" * 60)
        print(f"Processed {count} Full-Resolution images.")
        print(f"Average PSNR: {total_psnr / count:.4f} dB")
        print(f"Average SSIM: {total_ssim / count:.4f}")
    else:
        print("❌ No matched pairs found!")
        print(f"Please check your path: {RESTORED_ROOT}")

if __name__ == "__main__":
    main()