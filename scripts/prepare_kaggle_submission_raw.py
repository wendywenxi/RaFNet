import os
import sys
import torch
import scipy.io
import numpy as np
import pandas as pd
import base64
from collections import OrderedDict

# ================================
# 导入 model
# ================================
sys.path.append('.')   
from RaFNet_arch import RaFNet


def array_to_base64string(x):
    array_bytes = x.tobytes()
    base64_bytes = base64.b64encode(array_bytes)
    base64_string = base64_bytes.decode('utf-8')
    return base64_string


# ================================
# 构建 model 并加载权重
# ================================
def load_frenet_model(ckpt_path, device='cuda'):

    # ===== 1. 构建网络=====
    net = RaFNet(
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

    
    if 'params' in checkpoint:
        state_dict = checkpoint['params']
    else:
        state_dict = checkpoint

   
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith('module.'):
            k = k[7:]
        new_state_dict[k] = v

    net.load_state_dict(new_state_dict, strict=True)

    print(f'Loaded FrENet weights from {ckpt_path}')
    return net



@torch.no_grad()
def my_raw_denoiser(x, model, device):
    """
    x: numpy array, shape (256, 256), float32, range [0,1]
    """

    
    inp = torch.from_numpy(x).unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
    inp = inp.to(device)

    
    out = model(inp)

    
    out = out.squeeze(0).squeeze(0).cpu().numpy()
    # print(f"out.dtype:{out.dtype}")
    # print(f"out.shape:{out.shape}")
    # print(f"out.max:{out.max}")
    # print(f"out.min:{out.min}")

  
    out = np.clip(out, 0.0, 1.0).astype(np.float32)

    return out


def main():

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

 
    mat_path = '/datasets/SIDD_Medium_Raw/BenchmarkNoisyBlocksRaw.mat'
    key = 'BenchmarkNoisyBlocksRaw'
    ckpt_path = 'checkpoints/RaFNet_denoise.pth'
    output_csv = 'SubmitRaw.csv'

   
    model = load_frenet_model(ckpt_path, device)

  
    inputs = scipy.io.loadmat(mat_path)[key]
    print(f'inputs.shape = {inputs.shape}, dtype = {inputs.dtype}')

    output_blocks_base64string = []

  
    for i in range(inputs.shape[0]):
        for j in range(inputs.shape[1]):
            noisy_block = inputs[i, j]  # (256,256)

            denoised_block = my_raw_denoiser(
                noisy_block, model, device
            )

            
            assert denoised_block.shape == noisy_block.shape
            assert denoised_block.dtype == np.float32

            out_block_base64string = array_to_base64string(denoised_block)
            output_blocks_base64string.append(out_block_base64string)

  
    df = pd.DataFrame()
    df['ID'] = np.arange(len(output_blocks_base64string))
    df['BLOCK'] = output_blocks_base64string

    df.to_csv(output_csv, index=False)
    print(f'Saved submission file to {output_csv}')
    print('Ready to upload to Kaggle.')


if __name__ == '__main__':
    main()
