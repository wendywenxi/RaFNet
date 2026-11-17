import scipy.io as sio
import numpy as np
import os

mat_path = "/datasets/SIDD_Medium_Raw/ValidationNoisyBlocksRaw.mat"   
save_root = "/datasets/SIDD_Medium_Raw/val/noisy"  
os.makedirs(save_root, exist_ok=True)

data = sio.loadmat(mat_path)
raw_blocks = data["ValidationNoisyBlocksRaw"]  

num_images, num_blocks, H, W = raw_blocks.shape

print("Loaded RAW blocks:", raw_blocks.shape)



for i in range(num_images):
    for j in range(num_blocks):
        raw_patch = raw_blocks[i, j]    

        raw_u16 = np.clip(raw_patch * 65535.0, 0, 65535).astype(np.uint16)

        filename = f"scene_{i:02d}_patch_{j:02d}.raw"
        save_path = os.path.join(save_root, filename)

        raw_u16.tofile(save_path)

print("noisy转换完成！所有 .raw 文件已保存到:", save_root)


# mat_path = "/datasets/SIDD_Medium_Raw/ValidationGtBlocksRaw.mat"  
# save_root = "/datasets/SIDD_Medium_Raw/val/gt"    
# os.makedirs(save_root, exist_ok=True)

# # 读取 .mat 文件
# data = sio.loadmat(mat_path)
# raw_blocks = data["ValidationGtBlocksRaw"]   

# num_images, num_blocks, H, W = raw_blocks.shape

# print("Loaded RAW blocks:", raw_blocks.shape)

# for i in range(num_images):
#     for j in range(num_blocks):
#         raw_patch = raw_blocks[i, j]    

#         raw_u16 = np.clip(raw_patch * 65535.0, 0, 65535).astype(np.uint16)

#         filename = f"scene_{i:02d}_patch_{j:02d}.raw"
#         save_path = os.path.join(save_root, filename)

#         raw_u16.tofile(save_path)

# print("gt转换完成！所有 .raw 文件已保存到:", save_root)
