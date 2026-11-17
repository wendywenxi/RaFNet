# Efficient RAW Image Restoration via Radially Symmetric Frequency Modulation


Wenxi Zhao†, Wenlong Jiao†, Ping Wang, Pengfei Zhu, Dongwei Ren

Tianjin University · † Equal contribution

***
>**Abstract**: RAW sensor data preserve linear intensity and high dynamic range, offering a valuable basis for image restoration before in-camera image signal processing. However, RAW images are affected by diverse degradations, including blur, sensor noise, and color filter array subsampling, which exhibit distinct spectral characteristics and are commonly addressed with task-specific models. We propose RaFNet, a compact convolution-Fourier framework driven by radially symmetric adaptive frequency modulation for RAW image restoration, and evaluate it on RAW deblurring, denoising, and demosaicing. The framework integrates spatial- and frequency-domain processing through frequency-domain skip connections and Radial Frequency Blocks (RaF-Blocks). Each RaF-Block combines global spectral context modeling with local frequency modulation through the proposed Radially Symmetric Frequency Modulation (RSFM) module. Specifically, RSFM shares modulation parameters across frequency locations with the same spectral radius and rotates modulation kernels according to angular positions, thereby combining radial parameter sharing with directional sensitivity. We further analyze the proposed frequency modulation from a Wiener-filter perspective, providing theoretical insights into its underlying mechanism. Experiments on three RAW restoration benchmarks demonstrate that RaFNet achieves competitive restoration performance while maintaining a favorable trade-off between computational cost and model complexity.
***

## Installation

```bash
conda create -n rafnet python=3.11 -y
conda activate rafnet
pip install -r requirements.txt
```

## Dataset Preparation

Download datasets from their original providers and follow their usage terms. Dataset files are not included in this repository.

### RAW Deblurring: Deblur-RAW

Obtain the dataset from the [Deblur-RAW](https://github.com/bob831009/raw_image_deblurring).

The `PairedRawImageDataset` loader expects paired DNG files with matching filenames under session directories:

```text
DEBLUR_RAW/
├── train/
│   └── <session>/
│       ├── blur_raw/<name>.dng
│       └── sharp_raw/<name>.dng
└── test/
    └── <session>/
        ├── blur_raw/<name>.dng
        └── sharp_raw/<name>.dng
```

### RAW Denoising: SIDD

Obtain the RAW data from the [SIDD](https://abdokamel.github.io/sidd/). The manuscript uses SIDD Medium for training and the official benchmark for online evaluation.

1. In `scripts/data_preparation/sidd-raw.py`, set `input_folder` and both `save_folder` paths. Run:

   ```bash
   python -m scripts.data_preparation.sidd-raw
   ```

2. In `scripts/data_preparation/sidd-raw-val.py`, set the validation MAT path and output directory. Run it for the noisy/ground-truth blocks:

   ```bash
   python -m scripts.data_preparation.sidd-raw-val
   ```

Expected layout:

```text
SIDD_Medium_Raw/
├── train/
│   ├── input_crops/<name>.raw
│   └── gt_crops/<name>.raw
└── val/
    ├── noisy/<name>.raw
    └── gt/<name>.raw
```

### RAW Demosaicing: NTIRE 2025 RAIM

Use the Phase 1 paired data described in the [NTIRE 2025 RAIM dataset/challenge](https://drive.google.com/drive/folders/1Mgqve-yNcE26IIieI8lMIf-25VvZRs_J). The manuscript uses **38 training pairs and 12 test pairs** from 50 images, with 512 × 512 training patches extracted at stride 256.

Set `dataroot_lq` and `dataroot_gt` in the demosaicing YAML files:

```text
Demosaic/data/
├── train/
│   ├── lq/<name>.raw
│   └── gt/<name>.raw
└── val/
    ├── lq/<name>.raw
    └── gt/<name>.raw
```

## Training

### Single-GPU Training

```bash
# RAW deblurring
python -m basicsr.train -opt options/train/DeblurRaw/rafnet.yml

# RAW denoising
python -m basicsr.train -opt options/train/Denoise/rafnet-sidd.yml

# RAW demosaicing with DCRH (RaFNet+)
python -m basicsr.train -opt options/train/Demosaic/rafnet.yml
```

### Multi-GPU Training

For four-GPU distributed training, set `num_gpu: 4` and use, for example:

```bash
torchrun --standalone --nproc_per_node=4 --module basicsr.train \
  -opt options/train/DeblurRaw/rafnet.yml --launcher pytorch
```

Replace the configuration path to train another task.

### Training Settings

The supplied training configurations use the following settings:

| Task | Network class | Input → output channels | Training crop* | Batch per GPU | Iterations |
| :--- | :--- | :--- | ---: | ---: | ---: |
| Deblurring | `RaFNet` | 4 → 4 | 128 | 32 | 300,000 |
| Denoising | `RaFNet` | 1 → 1 | 256 | 8 | 400,000 |
| Demosaicing | `RaFNet_DCRH` | 1 → 3 | 128 | 16 | 400,000 |

*Crop size is `datasets.train.gt_size` before Bayer packing; deblurring packing halves each spatial dimension. With distributed training, the effective batch is the per-GPU batch multiplied by the process count.

All supplied training YAML files use:

- **Optimizer:** AdamW.
- **Initial learning rate:** `1e-3`.
- **Scheduler:** cosine annealing.
- **Loss:** `L1Loss + 0.1 × FFTLoss`.

## Evaluation

### Load Trained Weights

The test YAML files currently set `path.pretrain_network_g: ~`, which leaves the network randomly initialized. Set this field to a trained checkpoint before evaluating.

No pretrained checkpoint download links are currently provided in the repository or GitHub Releases.

For each task, update the dataset paths, GPU count, and checkpoint path:

```yaml
path:
  pretrain_network_g: /absolute/path/to/net_g_checkpoint.pth
  strict_load_g: true
```

### Task-Specific Configuration

**Deblurring**

In `options/test/DeblurRaw/rafnet.yml`, change `network_g.type: RAFNet` to `network_g.type: RaFNet` to match the implemented class.

For full-resolution sliding-window inference, enable:

```yaml
network_g:
  type: RaFNet
  # Retain the other network settings in the supplied YAML.
  train_size: 128
  grid_overlap_size: 64
```

**Demosaicing**

The supplied train/test YAML files use `RaFNet_DCRH`, corresponding to **RaFNet+**. For the base RaFNet variant, use `network_g.type: RaFNet` in both configurations and train/load matching weights.

The supplied test YAML uses 128 × 128 windows with 64-pixel overlap. To follow the manuscript's stated test protocol, update:

```yaml
val:
  # Retain the other validation settings in the supplied YAML.
  grids: true
  crop_size_h: 256
  crop_size_w: 256
  overlap_h: 128
  overlap_w: 128
```

**Denoising**

The supplied test YAML is configured for paired local validation and disables metric computation. Enable the PSNR/SSIM metric section when evaluating paired validation data.

Official benchmark results require the official online evaluation workflow.

### Run Evaluation

After applying the configuration changes:

```bash
# RAW deblurring
python -m basicsr.test -opt options/test/DeblurRaw/rafnet.yml

# RAW denoising
python -m basicsr.test -opt options/test/Denoise/rafnet-sidd.yml

# RAW demosaicing
python -m basicsr.test -opt options/test/Demosaic/rafnet.yml
```

Test logs and saved visualizations are written under `results/<name>/`. Saved visualizations are not a replacement for the RAW benchmark submission format.

## Results

Results reported in the manuscript:

| Task | Dataset / protocol | Model | PSNR ↑ | SSIM ↑ | MACs (G) ↓ | Params (M) ↓ |
| :--- | :--- | :--- | ---: | ---: | ---: | ---: |
| RAW deblurring | Deblur-RAW | RaFNet | 45.16 | 0.994 | 2.20 | 19.80 |
| RAW denoising | SIDD Benchmark, official online evaluation | RaFNet | 52.06 | 0.9930 | 35.03 | 19.80 |
| RAW demosaicing | NTIRE 2025 RAIM, manuscript 38/12 split | RaFNet | 34.74 | 0.9262 | 8.77 | 19.80 |
| RAW demosaicing | NTIRE 2025 RAIM, manuscript 38/12 split | RaFNet+ | 34.80 | 0.9287 | 9.18 | 19.82 |

RaFNet+ incorporates the Demosaicing Color Refinement Head (DCRH). MACs depend on the task and input resolution; values above follow the corresponding manuscript tables.



## Acknowledgements

The training and evaluation framework builds on [BasicSR](https://github.com/XPixelGroup/BasicSR) and [NAFNet](https://github.com/megvii-research/NAFNet). We thank the authors of these projects and the dataset providers.

## Contact

For implementation questions, please open a [GitHub issue](https://github.com/wendywenxi/RaFNet/issues).

- Wenxi Zhao: [wendyzhao@tju.edu.cn](mailto:wendyzhao@tju.edu.cn)
- Wenlong Jiao: [wenlongo@tju.edu.cn](mailto:wenlong@tju.edu.cn)
- Corresponding author, Dongwei Ren: [rendw@tju.edu.cn](mailto:rendw@tju.edu.cn)
