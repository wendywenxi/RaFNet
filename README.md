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

```bash
# RAW deblurring
torchrun --nproc_per_node=1 basicsr/train.py -opt options/train/DeblurRaw/rafnet.yml --launcher pytorch

# RAW denoising
torchrun --nproc_per_node=1 basicsr/train.py -opt options/train/Denoise/rafnet-sidd.yml --launcher pytorch

# RAW demosaicing 
torchrun --nproc_per_node=1 basicsr/train.py -opt options/train/Demosaic/rafnet.yml --launcher pytorch
```

### Training Settings

The supplied training configurations use the following settings:

| Task | Network class | Input → output channels | Training crop | Batch per GPU | Iterations |
| :--- | :--- | :--- | ---: | ---: | ---: |
| Deblurring | `RaFNet` | 4 → 4 | 128 | 32 | 300,000 |
| Denoising | `RaFNet` | 1 → 1 | 256 | 8 | 400,000 |
| Demosaicing | `RaFNet` | 1 → 3 | 128 | 16 | 400,000 |
| Demosaicing | `RaFNet_DCRH` | 1 → 4 | 128 | 16 | 400,000 |

## Evaluation

Trained models are stored in the `checkpoint/` directory. Before evaluation, update the dataset paths and set `path.pretrain_network_g` in the corresponding test YAML file.

### RAW Deblurring

```bash
python -m basicsr.test -opt options/test/DeblurRaw/rafnet.yml
```

### RAW Denoising

Denoising performance is evaluated on the SIDD RAW benchmark through the Kaggle platform.

Use the following script to organize the restored RAW predictions into the format required by Kaggle:

```bash
python scripts/prepare_kaggle_submission_raw.py
```

Upload the generated submission file to the SIDD RAW benchmarks on Kaggle for [PSNR evaluation](https://www.kaggle.com/competitions/sidd-benchmark-raw-psnr/submissions) and [SSIM evaluation](https://www.kaggle.com/competitions/sidd-benchmark-raw-ssim/submissions).

### RAW Demosaicing

```bash
python -m basicsr.test -opt options/test/Demosaic/rafnet.yml
```
Evaluation logs and saved visualizations are written to `results/<name>/`.

## Results

Results reported in the manuscript:

| Task | Dataset / protocol | Model | PSNR ↑ | SSIM ↑ | MACs (G) ↓ | Params (M) ↓ |
| :--- | :--- | :--- | ---: | ---: | ---: | ---: |
| RAW deblurring | Deblur-RAW | RaFNet | 45.16 | 0.994 | 2.20 | 19.80 |
| RAW denoising | SIDD Benchmark, official online evaluation | RaFNet | 52.06 | 0.9930 | 35.03 | 19.80 |
| RAW demosaicing | NTIRE 2025 RAIM | RaFNet | 34.74 | 0.9262 | 8.77 | 19.80 |
| RAW demosaicing | NTIRE 2025 RAIM | RaFNet+ | 34.80 | 0.9287 | 9.18 | 19.82 |

RaFNet+ incorporates the Demosaicing Color Refinement Head (DCRH). MACs depend on the task and input resolution; values above follow the corresponding manuscript tables.



## Acknowledgements

The training and evaluation framework builds on [BasicSR](https://github.com/XPixelGroup/BasicSR) and [NAFNet](https://github.com/megvii-research/NAFNet). We thank the authors of these projects and the dataset providers.

## Contact

For implementation questions, please open a [GitHub issue](https://github.com/wendywenxi/RaFNet/issues).

- Wenxi Zhao: [wendyzhao@tju.edu.cn](mailto:wendyzhao@tju.edu.cn)
- Wenlong Jiao: [wenlong@tju.edu.cn](mailto:wenlong@tju.edu.cn)
- Corresponding author, Dongwei Ren: [rendw@tju.edu.cn](mailto:rendw@tju.edu.cn)
