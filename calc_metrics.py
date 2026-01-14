import os
import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr_loss
from skimage.metrics import structural_similarity as ssim_loss

# ================= 配置区域 (修改这里) =================

# 1. 具体的原图路径 (精确到 .png/.jpg 文件)
GT_PATH = "Data/Images/bear_grass.png"

# 2. 结果图片所在的文件夹 (脚本会计算这里面所有图片)
#RESTORED_DIR = "outputs/Bear_Grid_Search/SD3/src_bear_grass/tar_0"
RESTORED_DIR = "outputs/SD3_Restoration_01/SD3/src_bear_grass/tar_0"
#RESTORED_DIR = "outputs/FlowEdit_SD3_Denoise/SD3/src_bear_grass_painted/tar_0"
# 3. 是否只在 Y 通道(亮度)上计算？(学术标准通常为 True)
TEST_Y_CHANNEL = True

# ====================================================

def to_y_channel(img):
    """将图像从 BGR (OpenCV格式) 转换为 Y 通道"""
    img = img.astype(np.float32) / 255.
    if img.ndim == 3 and img.shape[2] == 3:
        img = 65.481 * img[..., 2] + 128.553 * img[..., 1] + 24.966 * img[..., 0] + 16.0
        img = img / 255.0
    return img

def calculate_psnr_ssim(img_gt, img_restored, test_y_channel=False):
    """计算一对图片的 PSNR 和 SSIM"""
    
    # 1. 尺寸对齐 (裁剪多余边缘)
    h, w = img_restored.shape[:2]
    h_gt, w_gt = img_gt.shape[:2]
    if h != h_gt or w != w_gt:
        min_h, min_w = min(h, h_gt), min(w, w_gt)
        img_gt = img_gt[:min_h, :min_w]
        img_restored = img_restored[:min_h, :min_w]

    # 2. 转换 Y 通道或归一化
    if test_y_channel:
        img_gt = to_y_channel(img_gt)
        img_restored = to_y_channel(img_restored)
    else:
        img_gt = img_gt.astype(np.float32) / 255.
        img_restored = img_restored.astype(np.float32) / 255.

    # 3. 计算指标
    try:
        psnr_val = psnr_loss(img_gt, img_restored, data_range=1.0)
    except:
        psnr_val = 0

    try:
        # 兼容不同版本的 skimage
        if img_gt.ndim == 2:
            ssim_val = ssim_loss(img_gt, img_restored, data_range=1.0)
        else:
            ssim_val = ssim_loss(img_gt, img_restored, channel_axis=2, data_range=1.0)
    except:
        ssim_val = 0

    return psnr_val, ssim_val

def main():
    # 1. 读取原图 (Ground Truth)
    if not os.path.exists(GT_PATH):
        print(f"Error: 找不到原图: {GT_PATH}")
        return

    img_gt = cv2.imread(GT_PATH)
    if img_gt is None:
        print("Error: 原图读取失败 (可能文件损坏)")
        return

    print(f"Ground Truth: {os.path.basename(GT_PATH)}")
    print(f"Scanning Dir: {RESTORED_DIR}")
    print(f"Metric Mode:  {'Y-Channel' if TEST_Y_CHANNEL else 'RGB'}")
    print("-" * 70)
    print(f"{'Filename':<50} | {'PSNR':<8} | {'SSIM':<8}")
    print("-" * 70)

    # 2. 遍历结果文件夹
    if not os.path.exists(RESTORED_DIR):
        print(f"Error: 找不到结果文件夹: {RESTORED_DIR}")
        return

    files = [f for f in os.listdir(RESTORED_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    files.sort() # 排序，方便查看参数变化规律

    if len(files) == 0:
        print("文件夹里没有图片！")
        return

    best_psnr = 0
    best_file = ""

    for filename in files:
        restored_path = os.path.join(RESTORED_DIR, filename)
        img_restored = cv2.imread(restored_path)
        
        if img_restored is None:
            continue

        # 计算这一张图的指标
        psnr, ssim = calculate_psnr_ssim(img_gt, img_restored, test_y_channel=TEST_Y_CHANNEL)
        
        # 打印结果
        print(f"{filename:<50} | {psnr:.4f}   | {ssim:.4f}")

        # 记录最佳
        if psnr > best_psnr:
            best_psnr = psnr
            best_file = filename

    print("-" * 70)
    print(f"🏆 Best Result: {best_file}")
    print(f"   PSNR: {best_psnr:.4f}")

if __name__ == "__main__":
    main()