import os
import cv2
import numpy as np
from tqdm import tqdm

def add_gaussian_noise(img, sigma):
    """
    向图像添加高斯噪声
    :param img: 输入图像 (numpy array, uint8)
    :param sigma: 噪声标准差 (例如 15, 25, 50)
    :return: 加噪后的图像 (numpy array, uint8)
    """
    # 1. 将图像转换为 float32 进行计算，避免溢出
    img_float = img.astype(np.float32)
    
    # 2. 生成高斯噪声
    # np.random.normal(loc=均值, scale=标准差, size=形状)
    noise = np.random.normal(0, sigma, img_float.shape)
    
    # 3. 叠加噪声
    noisy_img = img_float + noise
    
    # 4. 截断像素值到 [0, 255] 范围 (Clip)
    # 因为加噪后可能会出现负数或超过255的数
    noisy_img = np.clip(noisy_img, 0, 255)
    
    # 5. 转回 uint8
    return noisy_img.astype(np.uint8)

def process_dataset():
    # --- 配置路径 ---
    src_dir = "Data/Kodak24"              # 源数据集路径
    dst_base_dir = "Data/Kodak24_noised"  # 输出根路径
    sigmas = [10, 25, 50]                 # 需要添加的噪声等级
    
    # 检查源目录是否存在
    if not os.path.exists(src_dir):
        print(f"错误: 找不到源目录 {src_dir}")
        return

    # 获取所有图片文件
    valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tif')
    image_files = [f for f in os.listdir(src_dir) if f.lower().endswith(valid_extensions)]
    
    if not image_files:
        print("未找到图片文件。")
        return

    print(f"找到 {len(image_files)} 张图片，开始处理...")

    # --- 开始循环处理 ---
    for sigma in sigmas:
        # 为每个 sigma 创建单独的子文件夹，例如 Data/Kodak24_noised/sigma_10
        save_dir = os.path.join(dst_base_dir, f"sigma_{sigma}")
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"\n正在生成 Sigma = {sigma} 的数据...")
        
        for img_name in tqdm(image_files):
            # 1. 读取原图
            img_path = os.path.join(src_dir, img_name)
            img = cv2.imread(img_path) # 默认为 BGR 格式
            
            if img is None:
                print(f"警告: 无法读取 {img_name}，跳过。")
                continue
            
            # 2. 添加噪声
            noisy_img = add_gaussian_noise(img, sigma)
            
            # 3. 保存图片
            # 保持原文件名，保存到对应的子文件夹
            save_path = os.path.join(save_dir, img_name)
            cv2.imwrite(save_path, noisy_img)
            
    print(f"\n所有处理完成！数据已保存至 {dst_base_dir}")

if __name__ == "__main__":
    # 为了保证噪声的可复现性，可以固定随机种子 (可选)
    np.random.seed(42) 
    process_dataset()