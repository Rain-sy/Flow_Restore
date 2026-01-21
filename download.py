import kagglehub
import shutil
import os
# 1. 先下载到默认缓存位置
print("正在下载...")
cache_path = kagglehub.dataset_download("drxinchengzhu/kodak24")
print("下载完成，缓存路径:", cache_path)

# 2. 指定你的目标路径
# 注意：'/Data' 指的是根目录下的 Data 文件夹。
# 如果你是想下载到当前脚本所在目录下的 Data 文件夹，请改为 './Data'
target_dir = "./Data" 

# 3. 将文件移动到目标路径
print(f"正在移动文件到 {target_dir} ...")

# 如果目标目录不存在，则创建
if not os.path.exists(target_dir):
    os.makedirs(target_dir)

# 使用 copytree 把数据复制过去 (dirs_exist_ok=True 允许覆盖)
# 也可以用 shutil.move，但 copytree 跨磁盘分区更稳定
shutil.copytree(cache_path, target_dir, dirs_exist_ok=True)

print(f"成功！数据集已保存在: {target_dir}")