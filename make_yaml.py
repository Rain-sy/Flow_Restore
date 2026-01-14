import os
import yaml

# ================= 配置 =================
# SIDD 数据集根目录 (请根据你的实际情况修改)
# 注意：指向包含很多 "0001_...", "0002_..." 文件夹的那一层
DATASET_ROOT = "Data/SIDD_Small_sRGB_Only/Data"

OUTPUT_YAML = "SIDD_dataset.yaml"

# 针对智能手机噪声的 Prompt
SOURCE_PROMPT = "noisy, grainy, smartphone camera noise, high ISO, low light artifacts, chroma noise"
TARGET_PROMPT = "clean, sharp, high quality, denoised, ground truth, crystal clear"
# =======================================

def generate_yaml():
    data_list = []
    
    print(f"Scanning SIDD dataset in: {DATASET_ROOT} ...")

    # 遍历所有子文件夹
    for root, dirs, files in os.walk(DATASET_ROOT):
        for file in files:
            # 只找噪声图 (SIDD 的命名通常包含 NOISY)
            if "NOISY" in file and file.lower().endswith(".png"):
                
                # 构造绝对路径或相对路径
                img_path = os.path.join(root, file).replace("\\", "/")
                
                # 获取父文件夹名 (作为唯一标识的一部分，虽然这里只生成 yaml，
                # 但配合刚才修改的 run_script.py 就能完美工作)
                folder_name = os.path.basename(root)
                
                entry = {
                    "input_img": img_path,
                    "source_prompt": SOURCE_PROMPT,
                    "target_prompts": [TARGET_PROMPT],
                    # 使用 文件夹名 作为 target code，方便追踪
                    "target_codes": [f"{folder_name}_restored"]
                }
                data_list.append(entry)

    # 保存
    with open(OUTPUT_YAML, 'w') as f:
        yaml.dump(data_list, f, sort_keys=False, default_flow_style=False)
    
    print(f"Done! Found {len(data_list)} images. Saved to {OUTPUT_YAML}")

if __name__ == "__main__":
    generate_yaml()