import yaml
import numpy as np

# ================= 配置区域 =================
OUTPUT_FILENAME = "grid_search.yaml"

# 循环范围 (1.5 到 5.0，步长 0.5)
# np.arange 的右边界是不包含的，所以写 5.1
src_range = np.arange(2, 10.1, 2)
tar_range = np.arange(2, 10.1, 2)

# 固定参数 (基于之前的最佳实践)
BASE_CONFIG = {
    "exp_name": "Bear_Grid_Search",    # 所有图会保存在 outputs/Bear_Grid_Search 下
    "dataset_yaml": "single_restore.yaml", # 指向你的 bear_grass 单图配置
    
    # === 模型设置 (如果是 FLUX 请修改这里) ===
    "model_type": "SD3",
    "sampler_type": "FlowEditSD3",
    "T_steps": 30,
    
    # === 结构参数 (固定) ===
    "coupling_strength": 0.75, # 强力锁死结构
    "n_min": 0,                # 只做最后微调
    "n_max": 23,
    "n_avg": 1,
    "seed": 42
}
# ===========================================

def generate_grid():
    experiment_list = []
    
    print(f"Generating grid search for SRC: {src_range} and TAR: {tar_range}")
    
    for src in src_range:
        for tar in tar_range:
            # 复制一份基础配置
            config = BASE_CONFIG.copy()
            
            # 设置动态变化的参数
            # 注意：必须转为 python float，否则 yaml 保存 numpy float 会很难看
            config["src_guidance_scale"] = round(float(src), 2)
            config["tar_guidance_scale"] = round(float(tar), 2)
            
            experiment_list.append(config)
            
    # 保存为 YAML
    with open(OUTPUT_FILENAME, "w", encoding='utf-8') as f:
        yaml.dump(experiment_list, f, sort_keys=False, default_flow_style=False)
        
    print(f"Done! Generated {len(experiment_list)} experiments in '{OUTPUT_FILENAME}'")

if __name__ == "__main__":
    generate_grid()