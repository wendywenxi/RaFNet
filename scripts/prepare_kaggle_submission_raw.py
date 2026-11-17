import os
import sys
import torch
import scipy.io
import numpy as np
import pandas as pd
import base64
from collections import OrderedDict

# ================================
# 【新增】导入 FrENet
# ================================
sys.path.append('.')   # 确保能 import FrENet_arch
# from FrENet_sharew_rotation_arch import FrENetShareW2
# from FrENet_sharew_rotation_nofftskip_arch import FrENetShareW2_nofftskip
from FrENet_sharew_rotation_wo_rsafpm_arch import FrENetShareW2_woRSAFPM


# ================================
# 官方工具函数（保持不变）
# ================================
def array_to_base64string(x):
    array_bytes = x.tobytes()
    base64_bytes = base64.b64encode(array_bytes)
    base64_string = base64_bytes.decode('utf-8')
    return base64_string


# ================================
# 【新增】构建 FrENet 并加载权重
# ================================
def load_frenet_model(ckpt_path, device='cuda'):

    # ===== 1. 构建网络（参数必须与你训练时一致）=====
    net = FrENetShareW2_woRSAFPM(
        in_channel=1,
        out_channel=1,
        width=32,
        middle_blk_num=8,
        enc_blk_nums=[2, 2, 4],
        dec_blk_nums=[4, 2, 2],
        img_size=256,
        fft_kernel_mode='fixed',
        fixed_fft_kernel=4,
    )

    net = net.to(device)
    net.eval()

    # ===== 2. 加载权重 =====
    checkpoint = torch.load(ckpt_path, map_location=device)

    # BasicSR 风格
    if 'params' in checkpoint:
        state_dict = checkpoint['params']
    else:
        state_dict = checkpoint

    # 去掉多余前缀
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith('module.'):
            k = k[7:]
        new_state_dict[k] = v

    net.load_state_dict(new_state_dict, strict=True)

    print(f'Loaded FrENet weights from {ckpt_path}')
    return net


# ================================
# 【修改】真正的 RAW 去噪函数
# ================================
@torch.no_grad()
def my_raw_denoiser(x, model, device):
    """
    x: numpy array, shape (256, 256), float32, range [0,1]
    """

    # -------- numpy → torch --------
    inp = torch.from_numpy(x).unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
    inp = inp.to(device)

    # -------- FrENet 推理 --------
    out = model(inp)

    # -------- torch → numpy --------
    out = out.squeeze(0).squeeze(0).cpu().numpy()
    # print(f"out.dtype:{out.dtype}")
    # print(f"out.shape:{out.shape}")
    # print(f"out.max:{out.max}")
    # print(f"out.min:{out.min}")

    # 安全裁剪（推荐）
    out = np.clip(out, 0.0, 1.0).astype(np.float32)

    return out


# ================================
# 主流程
# ================================
def main():

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # -------- 路径设置 --------
    mat_path = '/mnt/hdd/zwx/datasets/SIDD_Medium_Raw/BenchmarkNoisyBlocksRaw.mat'
    key = 'BenchmarkNoisyBlocksRaw'
    ckpt_path = '/mnt/hdd/zwx/frenet/experiments/FrENet-SIDD_Medium_Raw-width32-rotation-norsafpm/models/net_g_220000.pth'
    output_csv = 'SubmitRaw_rotaion_norsafpm_220000.csv'

    # -------- 加载模型 --------
    model = load_frenet_model(ckpt_path, device)

    # -------- 读取 Benchmark 数据 --------
    inputs = scipy.io.loadmat(mat_path)[key]
    print(f'inputs.shape = {inputs.shape}, dtype = {inputs.dtype}')

    output_blocks_base64string = []

    # -------- 推理 --------
    for i in range(inputs.shape[0]):
        for j in range(inputs.shape[1]):
            noisy_block = inputs[i, j]  # (256,256)

            denoised_block = my_raw_denoiser(
                noisy_block, model, device
            )

            # 评测接口要求：shape & dtype 完全一致
            assert denoised_block.shape == noisy_block.shape
            assert denoised_block.dtype == np.float32

            out_block_base64string = array_to_base64string(denoised_block)
            output_blocks_base64string.append(out_block_base64string)

    # -------- 保存 CSV --------
    df = pd.DataFrame()
    df['ID'] = np.arange(len(output_blocks_base64string))
    df['BLOCK'] = output_blocks_base64string

    df.to_csv(output_csv, index=False)
    print(f'Saved submission file to {output_csv}')
    print('Ready to upload to Kaggle.')


if __name__ == '__main__':
    main()
