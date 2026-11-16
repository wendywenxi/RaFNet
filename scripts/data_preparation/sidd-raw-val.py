import scipy.io as sio
import numpy as np
import os

mat_path = "/mnt/hdd/zwx/datasets/SIDD_Medium_Raw/ValidationNoisyBlocksRaw.mat"   # 改成你的 mat 路径
save_root = "/mnt/hdd/zwx/datasets/SIDD_Medium_Raw/val/noisy"    # 输出目录
os.makedirs(save_root, exist_ok=True)

# 读取 .mat 文件
data = sio.loadmat(mat_path)
raw_blocks = data["ValidationNoisyBlocksRaw"]   # shape: (40, 32, 256, 256)
# raw_blocks = data["BenchmarkNoisyBlocksRaw"]   # shape: (40, 32, 256, 256)

num_images, num_blocks, H, W = raw_blocks.shape

print("Loaded RAW blocks:", raw_blocks.shape)
# print("RAW blocks dtype:", raw_blocks.dtype)

# print(data.keys())
# raw_data = data['BenchmarkNoisyBlocksRaw']
# print(type(raw_data))
# print(raw_data.shape)
# print(raw_data.dtype)
# print(raw_data.min(), raw_data.max())


# 遍历全部 scene 和 patch
for i in range(num_images):
    for j in range(num_blocks):
        raw_patch = raw_blocks[i, j]     # 256×256 float32

        # 转为 uint16 raw 格式
        raw_u16 = np.clip(raw_patch * 65535.0, 0, 65535).astype(np.uint16)

        # 保存文件名
        filename = f"scene_{i:02d}_patch_{j:02d}.raw"
        save_path = os.path.join(save_root, filename)

        # 以二进制形式写入 .raw
        raw_u16.tofile(save_path)

print("noisy转换完成！所有 .raw 文件已保存到:", save_root)


# mat_path = "/mnt/hdd/zwx/datasets/SIDD_Medium_Raw/ValidationGtBlocksRaw.mat"   # 改成你的 mat 路径
# save_root = "/mnt/hdd/zwx/datasets/SIDD_Medium_Raw/val/gt"    # 输出目录
# os.makedirs(save_root, exist_ok=True)

# # 读取 .mat 文件
# data = sio.loadmat(mat_path)
# raw_blocks = data["ValidationGtBlocksRaw"]   # shape: (40, 32, 256, 256)

# num_images, num_blocks, H, W = raw_blocks.shape

# print("Loaded RAW blocks:", raw_blocks.shape)

# # 遍历全部 scene 和 patch
# for i in range(num_images):
#     for j in range(num_blocks):
#         raw_patch = raw_blocks[i, j]     # 256×256 float32

#         # 转为 uint16 raw 格式
#         raw_u16 = np.clip(raw_patch * 65535.0, 0, 65535).astype(np.uint16)

#         # 保存文件名
#         filename = f"scene_{i:02d}_patch_{j:02d}.raw"
#         save_path = os.path.join(save_root, filename)

#         # 以二进制形式写入 .raw
#         raw_u16.tofile(save_path)

# print("gt转换完成！所有 .raw 文件已保存到:", save_root)
