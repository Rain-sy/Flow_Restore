import os
from PIL import Image, ImageDraw

# ================= 配置区域 =================
# 输入：单张图片的具体路径
INPUT_PATH = "Data/Images/bear_grass.png"

# 输出：生成图片的文件夹
OUTPUT_DIR = "Data/Masked_Images"

# 黑块的大小 (像素)
MASK_SIZE = 200  
# ===========================================

def create_single_mask():
    # 1. 检查原图是否存在
    if not os.path.exists(INPUT_PATH):
        print(f"Error: 找不到文件 {INPUT_PATH}")
        return

    # 2. 创建输出文件夹
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # 3. 构造输出文件名 (保持同名)
    filename = os.path.basename(INPUT_PATH)
    save_path = os.path.join(OUTPUT_DIR, filename)

    try:
        # 4. 读取并转为 RGB (防止 4 通道报错)
        img = Image.open(INPUT_PATH)
        
        # 5. 计算中心位置
        w, h = img.size
        left = (w - MASK_SIZE) // 2
        top = (h - MASK_SIZE) // 2
        right = left + MASK_SIZE
        bottom = top + MASK_SIZE
        
        # 6. 画黑块
        draw = ImageDraw.Draw(img)
        draw.rectangle([left, top, right, bottom], fill=(0, 0, 0))
        
        # 7. 保存
        img.save(save_path)
        print(f"Success! Masked image saved to: {save_path}")
        
    except Exception as e:
        print(f"Error processing image: {e}")

if __name__ == "__main__":
    create_single_mask()