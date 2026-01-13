import os
import cv2
import numpy as np
import random

def add_gaussian_noise(image, sigma=25):
    noise = np.random.normal(0, sigma, image.shape)
    noisy_image = image.astype(np.float32) + noise
    noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
    
    return noisy_image

def process_folder(input_dir, output_dir, noise_level=25):
    # 1. 创建输出文件夹
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    # 2. 获取所有图片文件
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
    files = [f for f in os.listdir(input_dir) if os.path.splitext(f)[1].lower() in valid_extensions]
    
    print(f"Found {len(files)} images in {input_dir}. Adding Gaussian Noise (sigma={noise_level})...")

    # 3. 循环处理
    for idx, filename in enumerate(files):
        img_path = os.path.join(input_dir, filename)
        save_path = os.path.join(output_dir, filename)

        # 读取图片
        img = cv2.imread(img_path)
        if img is None:
            print(f"Skipping broken file: {filename}")
            continue

        # 添加噪声
        noisy_img = add_gaussian_noise(img, sigma=noise_level)

        # 保存图片
        cv2.imwrite(save_path, noisy_img)

        if (idx + 1) % 10 == 0:
            print(f"Processed {idx + 1}/{len(files)}")

    print("Done! All noisy images saved.")

if __name__ == "__main__":
    # ================= 配置区域 =================
    INPUT_DIR = "Data/Images"
    OUTPUT_DIR = "Data/Noisy_Images"
    
    # 噪声强度 (Sigma)
    # 15 = 轻微噪声
    # 25 = 中等噪声 (学术界去噪任务常用的标准)
    # 50 = 强噪声
    NOISE_SIGMA = 25
    if not os.path.exists(INPUT_DIR):
        print(f"Error: Input directory '{INPUT_DIR}' not found!")
    else:
        process_folder(INPUT_DIR, OUTPUT_DIR, NOISE_SIGMA)