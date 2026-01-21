import os
import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr_loss
from skimage.metrics import structural_similarity as ssim_loss

# ================= 配置区域 =================
# 1. 原始 Kodak24 GT 文件夹路径
GT_ROOT = "Data/Kodak24"

# 2. 包含各 Sigma 子文件夹的根目录
# 脚本会自动寻找该目录下的 sigma_10, sigma_25, sigma_50
NOISED_ROOT = "outputs/Kodak24_restored/SD3"

# 3. 需要评估的 Sigma 列表
SIGMA_LIST = [10, 25, 50]
# ===========================================

def calculate_metrics(img_gt, img_restored):
    """
    计算 PSNR 和 SSIM
    """
    # 1. 尺寸安全检查 (Cropping to overlap)
    h_res, w_res = img_restored.shape[:2]
    h_gt, w_gt = img_gt.shape[:2]
    
    if h_res != h_gt or w_res != w_gt:
        # print(f"   ⚠️ Size mismatch! GT: {w_gt}x{h_gt}, Res: {w_res}x{h_res}. Cropping...")
        min_h = min(h_res, h_gt)
        min_w = min(w_res, w_gt)
        img_gt = img_gt[:min_h, :min_w]
        img_restored = img_restored[:min_h, :min_w]

    # 2. 归一化到 [0, 1] 并不是必须的，skimage 可以处理 0-255，
    # 但为了和你的参考代码保持一致性，这里转为 uint8 或者 float 均可。
    # 这里直接使用 OpenCV 读取的 BGR uint8 数据计算，data_range=255
    
    # 注意：Kodak24 标准通常计算 RGB 平均 PSNR。
    # 如果你严格需要 Y 通道 (Luminance)，请取消下面注释的代码块。
    
    # --- Y 通道转换 (可选) ---
    # img_gt = img_gt.astype(np.float32) / 255.
    # img_restored = img_restored.astype(np.float32) / 255.
    # img_gt = 65.481 * img_gt[..., 2] + 128.553 * img_gt[..., 1] + 24.966 * img_gt[..., 0] + 16.0
    # img_restored = 65.481 * img_restored[..., 2] + 128.553 * img_restored[..., 1] + 24.966 * img_restored[..., 0] + 16.0
    # img_gt /= 255.0
    # img_restored /= 255.0
    # data_range = 1.0
    # -----------------------

    # --- 标准 RGB 计算 (Kodak 常用) ---
    data_range = 255
    # -------------------------------

    # 3. 计算 PSNR 和 SSIM
    # channel_axis=2 表示图片是 HWC 格式
    psnr = psnr_loss(img_gt, img_restored, data_range=data_range)
    ssim = ssim_loss(img_gt, img_restored, data_range=data_range, channel_axis=2)
    
    return psnr, ssim

def main():
    print(f"Dataset Root: {GT_ROOT}")
    print(f"Results Root: {NOISED_ROOT}")
    print("=" * 60)

    # 遍历每个 Sigma 设置
    for sigma in SIGMA_LIST:
        sigma_dir = f"sigma_{sigma}"
        restored_dir = os.path.join(NOISED_ROOT, sigma_dir)
        
        if not os.path.exists(restored_dir):
            print(f"❌ Skipping {sigma_dir}: Folder not found.")
            continue

        total_psnr = 0
        total_ssim = 0
        count = 0
        
        # 获取 GT 文件列表
        gt_files = [f for f in os.listdir(GT_ROOT) if f.lower().endswith(('.png', '.jpg', '.bmp'))]
        
        # 遍历每张图片
        for gt_file in gt_files:
            gt_path = os.path.join(GT_ROOT, gt_file)
            
            # 假设输出文件名和 GT 文件名一致
            restored_path = os.path.join(restored_dir, gt_file)
            
            if not os.path.exists(restored_path):
                # 尝试查找不同后缀的情况 (例如 GT是png, 结果是jpg)
                name_no_ext = os.path.splitext(gt_file)[0]
                candidates = [f for f in os.listdir(restored_dir) if f.startswith(name_no_ext)]
                if candidates:
                    restored_path = os.path.join(restored_dir, candidates[0])
                else:
                    # print(f"   Missing result for: {gt_file}")
                    continue

            # 读取图片 (BGR)
            img_gt = cv2.imread(gt_path)
            img_restored = cv2.imread(restored_path)

            if img_gt is None or img_restored is None:
                continue

            # 计算指标
            psnr, ssim = calculate_metrics(img_gt, img_restored)
            
            total_psnr += psnr
            total_ssim += ssim
            count += 1
        
        # 输出当前 Sigma 的结果
        if count > 0:
            avg_psnr = total_psnr / count
            avg_ssim = total_ssim / count
            print(f"[{sigma_dir}] Processed {count} images")
            print(f"   Average PSNR: {avg_psnr:.2f} dB   Average SSIM: {avg_ssim:.4f}")
            print("-" * 40)
        else:
            print(f"⚠️  [{sigma_dir}] No matched images found.")

if __name__ == "__main__":
    main()