import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import functools
import math

from torch.nn import init
            
class RaFNet(nn.Module):
    def __init__(self,
                 in_channel=4,
                 out_channel=4,
                 width=64,
                 middle_blk_num=4,
                 enc_blk_nums=[2, 2, 4],
                 dec_blk_nums=[4, 2, 2],
                 img_size=64,
                 train_size=False,
                 fft_kernel_mode='dynamic',
                 fixed_fft_kernel=4,
                 dynamic_fft_kernel_divisor=8,
                 grid_overlap_size=64
                 ) -> None:
        super(RaFNet, self).__init__()


        ##### val滑动窗口测试########
        self.train_size = train_size
        self.grid_overlap_size = (grid_overlap_size, grid_overlap_size) # ！滑动窗口重叠区域大小
        self.grid_kernel_size = [self.train_size, self.train_size] # 每个滑动窗口的空间尺寸
        self.grid = True
        ###################################

        ##### 动态或固定的fft_kernel########
        if fft_kernel_mode not in ['fixed', 'dynamic']:
            raise ValueError(f"fft_kernel_mode 必须是 'fixed' 或 'dynamic', 但得到了 '{fft_kernel_mode}'")
        self.fft_kernel_mode = fft_kernel_mode
        self.fixed_fft_kernel = fixed_fft_kernel
        self.dynamic_fft_kernel_divisor = dynamic_fft_kernel_divisor
        ###################################


        self.intro = nn.Conv2d(in_channels=in_channel, out_channels=width, kernel_size=3, padding=1, stride=1,
                               groups=1,
                               bias=True)
        self.ending = nn.Conv2d(in_channels=width, out_channels=out_channel, kernel_size=3, padding=1, stride=1,
                                groups=1,
                                bias=True)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        chan = width
        image_size = img_size

        # --- Encoder stages ---
        for num in enc_blk_nums:
            fft_kernel = self._get_fft_kernel(image_size)
            self.encoders.append(
                nn.ModuleList(
                    [RaFBlock(in_channel=chan, fft_kernel=fft_kernel, img_size=image_size) for _ in range(num)]
                )
            )
            self.downs.append(
                nn.Conv2d(in_channels=chan, out_channels=2 * chan, kernel_size=2, stride=2)
            )
            chan = 2 * chan
            image_size = image_size // 2

        # --- Middle blocks ---
        fft_kernel = self._get_fft_kernel(image_size)
        self.middle_blks = nn.ModuleList(
            [RaFBlock(in_channel=chan, fft_kernel=fft_kernel, img_size=image_size) for _ in range(middle_blk_num)]
        )

        # --- Decoder stages ---
        for num in dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False),
                    nn.PixelShuffle(2)
                )
            )
            chan = chan // 2
            image_size = image_size * 2
            fft_kernel = self._get_fft_kernel(image_size)
            self.decoders.append(
                nn.ModuleList(
                    [RaFBlock(in_channel=chan, fft_kernel=fft_kernel, img_size=image_size) for _ in range(num)]
                )
            )

    def _get_fft_kernel(self, current_image_size: int) -> int:
        """根据配置模式和当前图像大小计算 fft_kernel。"""
        if self.fft_kernel_mode == 'fixed':
            return self.fixed_fft_kernel
        # 'dynamic' 模式
        if current_image_size <= 8:
            # 对于非常小的特征图，使用较大的相对 kernel size
            return max(1, current_image_size // 2)
        else:
            return max(1, current_image_size // self.dynamic_fft_kernel_divisor)

    def grids(self, x):
        # print(f"x.shape: {x.shape}")
        b, c, h, w = x.shape
        self.original_size = (b, c, h, w)
        assert b == 1
        k1, k2 = self.grid_kernel_size
        k1 = min(h, k1)
        k2 = min(w, k2)
        grid_overlap_size = self.grid_overlap_size 

        stride = (k1 - grid_overlap_size[0], k2 - grid_overlap_size[1])
        self.stride = stride
        num_row = (h - grid_overlap_size[0] - 1) // stride[0] + 1 if stride[0] > 0 else 1
        num_col = (w - grid_overlap_size[1] - 1) // stride[1] + 1 if stride[1] > 0 else 1
        self.nr = num_row
        self.nc = num_col

        step_j = k2 if num_col == 1 else stride[1]
        step_i = k1 if num_row == 1 else stride[0]

        parts = []
        idxes = []
        i = 0 
        last_i = False

        while i < h and not last_i:
            j = 0
            if i + k1 >= h:
                i = h - k1
                last_i = True

            last_j = False
            while j < w and not last_j:
                if j + k2 >= w:
                    j = w - k2
                    last_j = True
                parts.append(x[:, :, i:i + k1, j:j + k2])
                idxes.append({'i': i, 'j': j})
                j = j + step_j
            i = i + step_i

        parts = torch.cat(parts, dim=0)
        self.idxes = idxes
        return parts

    def get_overlap_matrix(self, h, w, device, dtype):
        self.h = h
        self.w = w
        
        k1, k2 = self.grid_kernel_size
        k1, k2 = min(h, k1), min(w, k2)
        grid_overlap_size = self.grid_overlap_size
        stride = (k1 - grid_overlap_size[0], k2 - grid_overlap_size[1])
        
        num_row = (h - grid_overlap_size[0] - 1) // stride[0] + 1 if stride[0] > 0 else 1
        num_col = (w - grid_overlap_size[1] - 1) // stride[1] + 1 if stride[1] > 0 else 1
        self.nr, self.nc = num_row, num_col

        self.ek1 = max(0, self.nr * stride[0] + grid_overlap_size[0] * 2 - h)
        self.ek2 = max(0, self.nc * stride[1] + grid_overlap_size[1] * 2 - w)

        self.fuse_matrix_w1 = torch.linspace(1., 0., self.grid_overlap_size[1], device=device, dtype=dtype).view(1, 1, 1, self.grid_overlap_size[1])
        self.fuse_matrix_w2 = torch.linspace(0., 1., self.grid_overlap_size[1], device=device, dtype=dtype).view(1, 1, 1, self.grid_overlap_size[1])
        self.fuse_matrix_h1 = torch.linspace(1., 0., self.grid_overlap_size[0], device=device, dtype=dtype).view(1, 1, self.grid_overlap_size[0], 1)
        self.fuse_matrix_h2 = torch.linspace(0., 1., self.grid_overlap_size[0], device=device, dtype=dtype).view(1, 1, self.grid_overlap_size[0], 1)
        
        if self.ek2 > 0:
            self.fuse_matrix_ew1 = torch.linspace(1., 0., self.ek2, device=device, dtype=dtype).view(1, 1, 1, self.ek2)
            self.fuse_matrix_ew2 = torch.linspace(0., 1., self.ek2, device=device, dtype=dtype).view(1, 1, 1, self.ek2)
        else: 
            self.fuse_matrix_ew1 = torch.empty((1,1,1,0), device=device, dtype=dtype)
            self.fuse_matrix_ew2 = torch.empty((1,1,1,0), device=device, dtype=dtype)
        if self.ek1 > 0:
            self.fuse_matrix_eh1 = torch.linspace(1., 0., self.ek1, device=device, dtype=dtype).view(1, 1, self.ek1, 1)
            self.fuse_matrix_eh2 = torch.linspace(0., 1., self.ek1, device=device, dtype=dtype).view(1, 1, self.ek1, 1)
        else: 
            self.fuse_matrix_eh1 = torch.empty((1,1,0,1), device=device, dtype=dtype)
            self.fuse_matrix_eh2 = torch.empty((1,1,0,1), device=device, dtype=dtype)

    def grids_inverse(self, outs):
        outs_clone = outs.clone() 
        preds = torch.zeros(self.original_size, device=outs.device, dtype=outs.dtype)
        b_orig, c, h, w = self.original_size
        k1, k2 = self.grid_kernel_size
        k1, k2 = min(h, k1), min(w, k2)
        
        self.get_overlap_matrix(h, w, outs.device, outs.dtype) # 生成重叠融合的权重矩阵

        device = outs.device
        indices_i = torch.tensor([d['i'] for d in self.idxes], device=device, dtype=torch.long)
        indices_j = torch.tensor([d['j'] for d in self.idxes], device=device, dtype=torch.long)
        
        preds = grids_inverse_kernel(
            outs_clone, preds, indices_i, indices_j,
            self.fuse_matrix_h1, self.fuse_matrix_h2, self.fuse_matrix_eh1, self.fuse_matrix_eh2,
            self.fuse_matrix_w1, self.fuse_matrix_w2, self.fuse_matrix_ew1, self.fuse_matrix_ew2,
            k1, k2, h, w,
            self.grid_overlap_size[0], self.grid_overlap_size[1],
            self.ek1, self.ek2
        )
        return preds
    
    def forward(self, x):
        # print(f"[INPUT] x.shape = {x.shape}") # add
        # print(f"Train size: {self.train_size}, Grid overlap size: {self.grid_overlap_size}")
        BATCH, CHANNEL, HEIGHT, WIDTH = x.shape
        inp = x

        if self.grid and self.train_size: # 将大图像划分为重叠的网格块
            x = self.grids(x)
            # print('after grids:', x.shape) 


        # print(f"x.gridafter.shape: {x.shape}")

        x = self.intro(x)

        encs = [] # 空间跳跃连接
        x_fft_skips = [] # 频域跳跃连接

        current_x = x
        for stage_idx, (encoder_stage, down_layer) in enumerate(zip(self.encoders, self.downs)):
            stage_fft_skip = None
            for rafblock in encoder_stage:
                current_x, stage_fft_skip = rafblock(current_x, None, is_encoder=True)

            encs.append(current_x)
            x_fft_skips.append(stage_fft_skip)

            current_x = down_layer(current_x)

        middle_block_skip = 0
        for rafblock in self.middle_blks:
            current_x = rafblock(current_x, middle_block_skip, is_encoder=False)

        for stage_idx, (decoder_stage, up_layer, enc_skip, stage_input_fft_skip) in enumerate(zip(self.decoders, self.ups, encs[::-1],
                                                                                                  x_fft_skips[::-1])):
            current_x = up_layer(current_x)
            current_x = current_x + enc_skip

            current_stage_fft_skip = stage_input_fft_skip
            for block_idx, rafblock in enumerate(decoder_stage):
                current_x = rafblock(current_x, current_stage_fft_skip, is_encoder=False,)

        x = self.ending(current_x)
        
        if self.grid and self.train_size: # 将处理后的网格块拼接回完整图像
            x = self.grids_inverse(x)

   
        x = x[:, :, :HEIGHT, :WIDTH].contiguous() + inp

        # print(f"[OUTPUT] x.shape = {x.shape}") # add

        return x

# -----------------------------------------------------------------------

# for unet
class RaFBlock(nn.Module):
    def __init__(self, in_channel, fft_kernel=4, img_size=64, c_expand=2, ffn_expand=2):
        super(RaFBlock, self).__init__()

        self.c_expand = c_expand
        self.ffn_expand = ffn_expand
        channel_expand = self.c_expand * in_channel
        channel_ffn = self.ffn_expand * in_channel

        self.fft_kernel = fft_kernel

        # fft conv
        self.conv_fft1 = nn.Conv2d(in_channels=in_channel*2, out_channels=channel_expand*2, kernel_size=1, stride=1, padding=0, bias=True)
        self.conv_fft2 = nn.Conv2d(in_channels=channel_expand*2, out_channels=channel_expand*2, kernel_size=3, stride=1, padding=1, groups=channel_expand*2, bias=True)
        self.conv_fft3 = nn.Conv2d(in_channels=channel_expand, out_channels=in_channel*2, kernel_size=1, stride=1, padding=0, bias=True)

        self.ffn = FeedForward(dim=in_channel, ffn_expansion_factor=2.66, bias=True)

        # oca
        self.afpm_fft = RSFM(in_channel=channel_expand, kernel_size=fft_kernel, H=img_size, W=img_size)
        self.sca = SCA(in_channel=channel_expand)

        self.norm2 = LayerNorm2d(in_channel)
        self.norm_fft1 = LayerNorm2d(in_channel*2)

        self.beta = nn.Parameter(torch.zeros((1, in_channel, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, in_channel, 1, 1)), requires_grad=True)

        self.gelu = nn.GELU()
        self.sg = SimpleGate()

    def forward(self, x, x_fft_skip=None, is_encoder=False):
        batch, channel, height, width = x.shape
        x_size = [batch, channel, height, width]
        inp = x

        # FFT
        x_fft = torch.fft.fft2(x)
        x_fft = torch.fft.fftshift(x_fft, dim=(-2, -1))
        if is_encoder==False:
            x_fft = x_fft + x_fft_skip
        x_fft = torch.cat([x_fft.real, x_fft.imag], dim=1)
        x_fft = self.norm_fft1(x_fft)
        x_fft = self.conv_fft1(x_fft)
        x_fft = self.conv_fft2(x_fft)
        x_fft = self.sg(x_fft)
        local_fea = self.afpm_fft(x_fft)
        global_fea = self.sca(x_fft)
        x_fft = local_fea + global_fea
        x_fft_real, x_fft_imag = self.conv_fft3(x_fft).chunk(2, dim=1)
        x_fft = torch.complex(x_fft_real, x_fft_imag)
        if is_encoder:
            x_fft_skip = x_fft
        x_fft = torch.fft.ifftshift(x_fft, dim=(-2, -1))
        x_fft = torch.fft.ifft2(x_fft).real

        x_out = x_fft
        x_out = x_out * self.beta + inp

        inp = x_out

        # LayerNorm
        x_out_norm = self.norm2(x_out)
        x_ffn = self.ffn(x_out_norm)
        x_out = x_ffn
        x_out = x_out * self.gamma + inp

        if is_encoder:
            return x_out, x_fft_skip
        else:
            return x_out
        
        


#--------------------------------------------------------------------------

class LayerNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
            dim=0), None

class LayerNorm2d(nn.Module):

    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)

class SCA(nn.Module):
    def __init__(self, in_channel):
        super(SCA, self).__init__()
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=in_channel, out_channels=in_channel, kernel_size=1),
        )

    def forward(self, x):
        y = self.sca(x)
        return x * y


class RSFM(nn.Module):
    def __init__(self, in_channel, kernel_size, H=64, W=64, mlp_hidden_dim=None):
        super().__init__()
        # --- __init__ is IDENTICAL to your original code ---
        self.in_channel = in_channel
        self.K = kernel_size
        self.H, self.W = H, W 
        if H % self.K != 0 or W % self.K != 0: pass
        self.num_h, self.num_w, self.L = H // self.K, W // self.K, (H // self.K) * (W // self.K)
        center_y, center_x = H / 2.0, W / 2.0
        max_dist = math.sqrt(center_y**2 + center_x**2) or 1.0
        patch_positions = torch.zeros(self.L)
        for l in range(self.L):
            i, j = l // self.num_w, l % self.num_w
            patch_center_y, patch_center_x = i * self.K + self.K / 2.0, j * self.K + self.K / 2.0
            dist = math.sqrt((patch_center_y - center_y)**2 + (patch_center_x - center_x)**2)
            patch_positions[l] = dist / max_dist
        self.register_buffer('patch_pos_dist', patch_positions)
        if mlp_hidden_dim is None: mlp_hidden_dim = in_channel
        self.kernel_generator_mlp = nn.Sequential(nn.Linear(1, mlp_hidden_dim), nn.GELU(), nn.Linear(mlp_hidden_dim, self.K * self.K))
        self.pos_bias_mlp = nn.Sequential(nn.Linear(1, mlp_hidden_dim), nn.GELU(), nn.Linear(mlp_hidden_dim, 1))
        self.global_conv = nn.Conv2d(in_channels=in_channel, out_channels=in_channel, kernel_size=1, groups=1)
        self.pos_cache = {}

    def _get_dynamic_pos_encoding(self, H, W, device):
        if H % self.K != 0 or W % self.K != 0: raise ValueError(f"[RSFM Dynamic] Input dimensions H={H}, W={W} must be divisible by kernel_size={self.K}.")
        num_h, num_w = H // self.K, W // self.K
        L = num_h * num_w
        center_y, center_x = H / 2.0, W / 2.0
        max_dist = math.sqrt(center_y**2 + center_x**2) or 1.0
        patch_positions = torch.zeros(L, device=device)

        patch_angles = torch.zeros(L, device=device)

        for l in range(L):
            i, j = l // num_w, l % num_w
            patch_center_y, patch_center_x = i * self.K + self.K / 2.0, j * self.K + self.K / 2.0
            dy = patch_center_y - center_y
            dx = patch_center_x - center_x
            dist = math.sqrt(dy**2 + dx**2)
            
            angle = math.atan2(dy, dx) # add
            patch_positions[l] = dist / max_dist
            patch_angles[l] = angle # add

        # 四舍五入到6位小数以处理浮点精度问题
        unique_distances, inverse_indices = torch.unique(
            torch.round(patch_positions * 1e6) / 1e6,
            return_inverse=True
        )

        unique_dist_input = unique_distances.unsqueeze(-1)  # [num_unique, 1]
        unique_pos_kernels = self.kernel_generator_mlp(unique_dist_input)  # [num_unique, K*K]
        unique_pos_bias = self.pos_bias_mlp(unique_dist_input)  # [num_unique, 1]
        
        # 根据映射关系，将唯一的kernel和bias扩展到所有L个位置
        pos_kernels = unique_pos_kernels[inverse_indices].contiguous()  # [L, K*K]
        pos_bias = unique_pos_bias[inverse_indices].contiguous()  # [L, 1]
       

        patch_angles = patch_angles.contiguous()

        final_kernels = self._batch_rotate_kernels(pos_kernels, patch_angles)
        result = (L, final_kernels, pos_bias)
        
        return result
    
    def _batch_rotate_kernels(self, kernels, angles, chunk_size=8192):
        """
        kernels: [L, K*K]
        angles:  [L]
        """
        L = kernels.shape[0]
        K = self.K

        if K == 1:
            return kernels

        # 保险起见，先保证连续
        kernels = kernels.contiguous()
        angles = angles.contiguous()

        # 用 reshape 比 view 更稳
        kernels_4d = kernels.reshape(L, 1, K, K)

        rotated_list = []

        # 只在这一步局部关闭 cuDNN，其他卷积仍然可以继续用 cuDNN
        with torch.backends.cudnn.flags(enabled=False):
            for start in range(0, L, chunk_size):
                end = min(start + chunk_size, L)

                ker = kernels_4d[start:end].contiguous()
                ang = angles[start:end].contiguous()

                theta = torch.zeros(
                    end - start, 2, 3,
                    device=ker.device,
                    dtype=ker.dtype
                )

                cos_a = torch.cos(ang).to(dtype=ker.dtype)
                sin_a = torch.sin(ang).to(dtype=ker.dtype)

                theta[:, 0, 0] = cos_a
                theta[:, 0, 1] = -sin_a
                theta[:, 1, 0] = sin_a
                theta[:, 1, 1] = cos_a

                grid = F.affine_grid(theta, ker.size(), align_corners=True).contiguous()

                rot = F.grid_sample(
                    ker,
                    grid,
                    mode='bilinear',
                    padding_mode='zeros',
                    align_corners=True
                )

                rotated_list.append(rot.reshape(end - start, K * K))

        return torch.cat(rotated_list, dim=0)
    
    def forward(self, x):
        B, C, H, W = x.shape
        K = self.K
        
        # --- 动态计算部分 (保持不变) ---
        L, pos_kernels, pos_bias = self._get_dynamic_pos_encoding(H, W, device=x.device)
        
        # --- 维度操作修正 ---
        # 1. Unfold and reshape for processing
        unfolded_x = F.unfold(x, K, stride=K) # Shape: (B, C * K*K, L)
        unfolded_x = unfolded_x.view(B, C, K*K, L) # Shape: (B, C, K*K, L)

        # 2. Einsum and bias addition
        # Permute for einsum: (B, C, K*K, L) -> (B, C, L, K*K)
        unfolded_x_for_einsum = unfolded_x.permute(0, 1, 3, 2)
        features = torch.einsum('bclk,lk->bcl', unfolded_x_for_einsum, pos_kernels) # Shape: (B, C, L)
        features = features + pos_bias.view(1, 1, L)
        
        # 3. Global convolution
        # Reshape for conv: (B, C, L) -> (B*L, C, 1, 1) -> (B, C, L)
        features = self.global_conv(features.permute(0, 2, 1).contiguous().view(B*L, C, 1, 1))
        features = features.view(B, L, C).permute(0, 2, 1).contiguous()
        
        # 4. Modulation
        # Reshape features to match unfolded_x for broadcasting: (B, C, L) -> (B, C, 1, L)
        features = features.unsqueeze(2) # Shape: (B, C, 1, L)
        modulated_4d = unfolded_x * features # Shape: (B, C, K*K, L)
        
        # 5. Reshape back to 3D for fold
        # (B, C, K*K, L) -> (B, C * K*K, L)
        modulated_3d = modulated_4d.view(B, C * K*K, L)

        # 6. Fold the 3D tensor
        output = F.fold(modulated_3d, (H, W), kernel_size=K, stride=K)

        return output


class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias, ffn='ffn', window_size=None):
        super(FeedForward, self).__init__()

        self.ffn_expansion_factor = ffn_expansion_factor

        self.ffn = ffn
        if self.ffn_expansion_factor == 0:
            hidden_features = dim
            self.project_in = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

            self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1,
                                    groups=dim, bias=bias)
        else:
            hidden_features = int(dim*ffn_expansion_factor)
            self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)
            self.act = nn.GELU()
            self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

        self.dim = dim
        self.hidden_dim = hidden_features
    def forward(self, inp):
        x = self.project_in(inp)
        if self.ffn_expansion_factor == 0:
            x = self.act(self.dwconv(x))
        else:
            x1, x2 = self.dwconv(x).chunk(2, dim=1)
            x = self.act(x1) * x2
        x = self.project_out(x)
        return x

@torch.jit.script
def grids_inverse_kernel(
    outs: torch.Tensor,
    preds: torch.Tensor,
    indices_i: torch.Tensor,
    indices_j: torch.Tensor,
    fuse_matrix_h1: torch.Tensor,
    fuse_matrix_h2: torch.Tensor,
    fuse_matrix_eh1: torch.Tensor,
    fuse_matrix_eh2: torch.Tensor,
    fuse_matrix_w1: torch.Tensor,
    fuse_matrix_w2: torch.Tensor,
    fuse_matrix_ew1: torch.Tensor,
    fuse_matrix_ew2: torch.Tensor,
    k1: int, k2: int, h: int, w: int,
    grid_overlap_size_h: int,
    grid_overlap_size_w: int,
    ek1: int, ek2: int
) -> torch.Tensor:
    
    for cnt in range(outs.shape[0]):
        i = indices_i[cnt].item()
        j = indices_j[cnt].item()
        
        part = outs[cnt:cnt+1] 

        if i != 0 and i + k1 != h:
            part[:, :, :grid_overlap_size_h, :] *= fuse_matrix_h2
        if i + k1 * 2 - ek1 < h:
            part[:, :, -grid_overlap_size_h:, :] *= fuse_matrix_h1
        if ek1 > 0 and i + k1 == h:
            part[:, :, :ek1, :] *= fuse_matrix_eh2
        if ek1 > 0 and i + k1 * 2 - ek1 == h:
            part[:, :, -ek1:, :] *= fuse_matrix_eh1

        if j != 0 and j + k2 != w:
            part[:, :, :, :grid_overlap_size_w] *= fuse_matrix_w2
        if j + k2 * 2 - ek2 < w:
            part[:, :, :, -grid_overlap_size_w:] *= fuse_matrix_w1

        if ek2 > 0 and j + k2 == w:
            part[:, :, :, :ek2] *= fuse_matrix_ew2
        if ek2 > 0 and j + k2 * 2 - ek2 == w:
            part[:, :, :, -ek2:] *= fuse_matrix_ew1
            
        preds[0, :, i:i + k1, j:j + k2] += part.squeeze(0)

    return preds

  
class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2
    