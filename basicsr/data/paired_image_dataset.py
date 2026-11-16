import rawpy
import numpy as np
from pathlib import Path
from torch.utils import data

from basicsr.data.data_util import (
    paired_paths_from_folder,
    Packing,
    Unpacking,
)
from basicsr.data.transforms import (
    augment,
    paired_random_crop,
    random_augmentation,
)
from basicsr.utils import FileClient, img2tensor, padding
        
class PairedRawImageDataset(data.Dataset):
    """Paired raw/rgb image dataset for image restoration within Basicsr framework.

    Reads LQ (e.g. blurry, noisy raw or rgb) and GT raw/rgb image pairs.
    Adapts user's raw reading and Packing/Unpacking logic.

    Args:
        opt (dict): Config for train/val datasets. It contains the following keys:
            root_dir (str): Root path containing session folders (e.g., 'path/to/your/dataset_root').
                            Each session folder must contain 'blur_raw' and 'sharp_raw' subfolders
                            with corresponding .dng files, and/or 'blur' and 'sharp' subfolders with .png files.
            data_domain (str): 'raw' or 'rgb'.
            max_pixel_value (float): Maximum pixel value for raw sensor data (e.g., 16383.0 for 14-bit).
            phase (str): 'train' or 'val'.
            gt_size (int): Cropped patched size for gt patches during training. Set to 0 or None to disable.
            crop_size (int): Cropped patched size for gt patches during validation (center crop). Set to 0 or None to disable.
            geometric_augs (bool): Use geometric augmentations (random flip, rotation) during training.
            scale (int): Scale factor. Usually 1 for denoising/deblurring, but required by basicsr structure.
    """

    def __init__(self, opt):
        super(PairedRawImageDataset, self).__init__()
        self.opt = opt
        self.root_dir = opt['root_dir']
        self.data_domain = opt.get('data_domain', 'raw').lower()  # 'raw' or 'rgb'
        self.max_pixel_value = float(opt.get('max_pixel_value', 16383.0)) # Ensure float division later
        self.phase = opt['phase']

        # Geometric augmentations for training
        if self.phase == 'train':
            self.gt_size = opt.get('gt_size', 0) # Default to 0 (no cropping)
            self.geometric_augs = opt.get('geometric_augs', False)
        elif self.phase == 'val':
            self.gt_size = opt.get('gt_size', 0) # Default to 0 (no cropping)
        else:
            self.gt_size = 0

        # Get image path pairs
        if self.data_domain == 'raw':
            self.paths = self._scan_raw_folders()
        elif self.data_domain == 'rgb':
            self.paths = self._scan_rgb_folders()
        else:
            raise ValueError(f"Unknown data_domain: {self.data_domain}, should be 'raw' or 'rgb'.")
        if not self.paths:
            raise ValueError(f"No paired files found in {self.root_dir} for domain {self.data_domain}.")

        print(f'Dataset: {self.__class__.__name__} - {len(self.paths)} image pairs found. Domain: {self.data_domain}')

    def _scan_raw_folders(self):
        """
        Scans the root directory for paired blur_raw and sharp_raw images (.dng).
        Assumes structure: root_dir/session_folder/{blur_raw, sharp_raw}/image_name.dng
        """
        pairs = []
        root_path = Path(self.root_dir)

        if not root_path.exists():
             print(f"Error: root_dir not found at {root_path}")
             return pairs

        for session_folder in root_path.iterdir():
            if session_folder.is_dir():
                blur_raw_path = session_folder / 'blur_raw'
                sharp_raw_path = session_folder / 'sharp_raw'

                if blur_raw_path.is_dir() and sharp_raw_path.is_dir():
                    blur_files = {f.name for f in blur_raw_path.iterdir() if f.suffix.lower() == '.dng' and f.is_file()}
                    sharp_files = {f.name for f in sharp_raw_path.iterdir() if f.suffix.lower() == '.dng' and f.is_file()}

                    common_files = blur_files.intersection(sharp_files)

                    for img_name in sorted(list(common_files)): # Sort for consistent order
                        lq_path = str(blur_raw_path / img_name)
                        gt_path = str(sharp_raw_path / img_name)
                        pairs.append({'lq_path': lq_path, 'gt_path': gt_path})

        return pairs

    def _scan_rgb_folders(self):
        """
        Scans the root directory for paired blur and sharp images (.png).
        Assumes structure: root_dir/session_folder/{blur, sharp}/image_name.png
        """
        from pathlib import Path
        pairs = []
        root_path = Path(self.root_dir)
        if not root_path.exists():
            print(f"Error: root_dir not found at {root_path}")
            return pairs
        for session_folder in root_path.iterdir():
            if session_folder.is_dir():
                blur_path = session_folder / 'blur'
                sharp_path = session_folder / 'sharp'
                if blur_path.is_dir() and sharp_path.is_dir():
                    blur_files = {f.name for f in blur_path.iterdir() if f.suffix.lower() == '.png' and f.is_file()}
                    sharp_files = {f.name for f in sharp_path.iterdir() if f.suffix.lower() == '.png' and f.is_file()}
                    common_files = blur_files.intersection(sharp_files)
                    for img_name in sorted(list(common_files)):
                        lq_path = str(blur_path / img_name)
                        gt_path = str(sharp_path / img_name)
                        pairs.append({'lq_path': lq_path, 'gt_path': gt_path})
        return pairs

    def _read_raw_image(self, path):
        """
        Reads a raw image (.dng) using rawpy and applies basic preprocessing.
        Returns a numpy array [H, W, 1] float32 in [0, 1].
        """
        try:
            with rawpy.imread(path) as raw:
                # Get black level (average across channels for simplicity, adjust if needed)
                black_level = np.mean(raw.black_level_per_channel)

                # Read visible raw image data
                image = raw.raw_image_visible.astype(np.float32)

                # Subtract black level
                image = image - black_level

                # Clip negative values resulting from black level subtraction
                image = np.clip(image, 0, None)

                # Normalize to [0, 1] using max pixel value and black level
                # Avoid division by zero if max_pixel_value equals black_level
                norm_range = self.max_pixel_value - black_level
                if norm_range <= 0:
                     print(f"Warning: Max pixel value ({self.max_pixel_value}) <= black level ({black_level}) for {path}. Normalization may fail.")
                     image = image / self.max_pixel_value if self.max_pixel_value > 0 else image * 0
                else:
                    image = image / norm_range

                # Add channel dimension [H, W] -> [H, W, 1] for compatibility with HWC-based augs
                image = np.expand_dims(image, axis=-1)

                return image # HWC, float32, [0, 1]

        except Exception as e:
            print(f"Error reading or processing raw image {path}: {e}")
            raise # Re-raise the exception after printing

    def _read_rgb_image(self, path):
        """
        Reads a png image using cv2 and returns a numpy array [H, W, 3] float32 in [0, 1].
        """
        import cv2
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError(f"Failed to read image: {path}")
        if img.ndim == 2:
            img = img[:, :, None]  # 灰度图
        elif img.shape[2] == 4:
            img = img[:, :, :3]  # 去掉alpha通道
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        return img  # HWC, float32, [0, 1]

    def __getitem__(self, index):
        index = index % len(self.paths) # Handle potential index wrapping if sampler is used

        lq_path = self.paths[index]['lq_path']
        gt_path = self.paths[index]['gt_path']

        # Load and preprocess raw images
        # Returns HWC numpy arrays (H, W, 1) or (H, W, 3), float32, range [0, 1]
        if self.data_domain == 'raw':
            img_lq = self._read_raw_image(lq_path)
            img_gt = self._read_raw_image(gt_path)
        elif self.data_domain == 'rgb':
            img_lq = self._read_rgb_image(lq_path)
            img_gt = self._read_rgb_image(gt_path)
        else:
            raise ValueError(f"Unknown data_domain: {self.data_domain}")

        
        # --- Apply Basicsr Geometric Augmentations (on HWC numpy) ---
        # Augmentations are applied on the [H, W, 1] numpy array.
        # Padding and cropping work correctly with the last dimension being 1.
        # Random flip/rotation also work on HWC data.

        if self.phase == 'train':
            gt_size = self.gt_size
            if gt_size > 0:
                # padding
                img_gt, img_lq = padding(img_gt, img_lq, gt_size)
                # random crop
                # Pass a dummy scale of 1 as it's not used in the core cropping logic for paired data.
                # The original basicsr function uses scale internally for coordinate mapping, but
                # for paired crop with scale 1, it's a direct mapping.
                if self.data_domain == 'raw':
                    img_gt = Packing(img_gt)
                    img_lq = Packing(img_lq)
                    img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size, 1, gt_path)
                    img_gt = Unpacking(img_gt)
                    img_lq = Unpacking(img_lq)
                else:
                    img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size, 1, gt_path)

            # flip, rotation augmentations
            if self.geometric_augs:
                # Random augmentation on HWC data
                img_gt, img_lq = random_augmentation(img_gt, img_lq)


        # --- Convert to Tensor, Apply Packing/Unpacking ---
        # HWC numpy ([H, W, 1] or [H, W, 3]) to CHW tensor ([1, H, W] or [3, H, W])
        # bgr2rgb=False because it's not a BGR image.
        img_gt_t, img_lq_t = img2tensor([img_gt, img_lq], bgr2rgb=False, float32=True)
        if self.data_domain == 'raw':
            lq = Packing(img_lq_t)
            gt = Packing(img_gt_t)
        else:
            lq = img_lq_t
            gt = img_gt_t

        # print(f"[DATA] lq shape: {lq.shape}, gt shape: {gt.shape}") # add

        return {
            'lq': lq,
            'gt': gt,
            'lq_path': lq_path,
            'gt_path': gt_path
        }

    def __len__(self):
        return len(self.paths)

    
class PairedDemosaicDataset(data.Dataset):
    def __init__(self, opt):
        super(PairedDemosaicDataset, self).__init__()
        self.opt = opt
        self.file_client = None
        self.io_backend_opt = opt['io_backend']
        
        self.gt_folder, self.lq_folder = opt['dataroot_gt'], opt['dataroot_lq']
        
        if 'filename_tmpl' in opt:
            self.filename_tmpl = opt['filename_tmpl']
        else:
            self.filename_tmpl = '{}'

        self.paths = paired_paths_from_folder(
            [self.lq_folder, self.gt_folder], ['lq', 'gt'],
            self.filename_tmpl) # 获取所有成对的lq和gt图像文件路径
        
        # ###################################
        # # 打印 paths 以便调试（只输出前30条）
        # print("[PairedDemosaicDataset] 加载paths（前30条）:")
        # try:
        #     for i, p in enumerate(self.paths):
        #         if i >= 30:
        #             print(f"...共{len(self.paths)}条，已省略后续...")
        #             break
        #         print(p)
        # except Exception as e:
        #     print("[PairedDemosaicDataset] 打印paths失败:", e)
        # ###################################

        if self.opt['phase'] == 'train':
            self.geometric_augs = opt.get('geometric_augs', True) # 是否数据几何增强

    def _read_raw_file(self, file_bytes, is_gt=False):
        img = np.frombuffer(file_bytes, dtype=np.uint16)
        
        if is_gt:
            dim = int(np.sqrt(img.size / 3))
            img = img.reshape((dim, dim, 3))
        else:
            dim = int(np.sqrt(img.size))
            img = img.reshape((dim, dim, 1))
            
        return img.astype(np.float32) / 65535.0

    def __getitem__(self, index):
        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt.get('scale', 1)
        gt_path = self.paths[index]['gt_path']
        lq_path = self.paths[index]['lq_path']
        
        img_gt_bytes = self.file_client.get(gt_path, 'gt')
        try:
            img_gt = self._read_raw_file(img_gt_bytes, is_gt=True)
        except Exception as e:
            raise Exception(f"GT path {gt_path} broken: {e}")

        img_lq_bytes = self.file_client.get(lq_path, 'lq')
        try:
            img_lq = self._read_raw_file(img_lq_bytes, is_gt=False)
        except Exception as e:
            raise Exception(f"LQ path {lq_path} broken: {e}")
        

        if self.opt['phase'] == 'train':
            gt_size = self.opt['gt_size']
            
            img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size, scale, gt_path)

            if self.geometric_augs:
                img_gt, img_lq = augment([img_gt, img_lq], self.opt['use_flip'], self.opt['use_rot'])

        # print("---- Debug dataset ----")
        # print("LQ:", lq_path, img_lq.shape, img_lq.dtype, img_lq.min(), img_lq.max())
        # print("GT:", gt_path, img_gt.shape, img_gt.dtype,
        # img_gt.min(), img_gt.max())
        # print("------------------------")

        img_gt, img_lq = img2tensor([img_gt, img_lq], bgr2rgb=False, float32=True)
        
        return {
            'lq': img_lq,
            'gt': img_gt,
            'lq_path': lq_path,
            'gt_path': gt_path
        }

    def __len__(self):
        return len(self.paths)

        
class PairedSIDDRawDataset(data.Dataset):
    """ Paired RAW dataset: both GT and LQ are .raw files.

    Raw files format assumed:
        - uint16 binary dump
        - LQ: 1 channel (H, W)
        - GT: 1 channel (H, W)
    """

    def __init__(self, opt):
        super(PairedSIDDRawDataset, self).__init__()
        self.opt = opt
        self.file_client = None
        self.io_backend_opt = opt['io_backend']

        self.mean = None
        self.std = None
        self.bgr2rgb = False   # RAW 不存在 BGR/RGB

        self.gt_folder = opt['dataroot_gt']
        self.lq_folder = opt['dataroot_lq']
        self.filename_tmpl = opt.get('filename_tmpl', '{}')
        
        self.paths = paired_paths_from_folder(
            [self.lq_folder, self.gt_folder], ['lq', 'gt'],
            self.filename_tmpl)

        if self.opt['phase'] == 'train':
            self.geometric_augs = opt.get('geometric_augs', True)

    # -------------------------------------------
    # 读取 .raw 文件（uint16），自动 reshape
    # -------------------------------------------
    def _read_raw_file(self, file_bytes, is_gt=False):

        raw = np.frombuffer(file_bytes, dtype=np.uint16)
        
        dim = int(np.sqrt(raw.size))
        img = raw.reshape(dim, dim, 1)

        img = img.astype(np.float32) / 65535.0 # 归一化。（原sidd数据已经过黑电平校正）

        return img

    # -------------------------------------------
    # 读取数据
    # -------------------------------------------
    def __getitem__(self, index):

        if self.file_client is None:
            self.file_client = FileClient(
                self.io_backend_opt.pop('type'), **self.io_backend_opt)

        scale = self.opt['scale']
        gt_path = self.paths[index]['gt_path']
        lq_path = self.paths[index]['lq_path']

        # Read GT RAW
        gt_bytes = self.file_client.get(gt_path, 'gt')
        try:
            img_gt = self._read_raw_file(gt_bytes, is_gt=True)
        except Exception as e:
            raise Exception(f"GT RAW corrupted: {gt_path}, err={e}")

        # Read LQ RAW
        lq_bytes = self.file_client.get(lq_path, 'lq')
        try:
            img_lq = self._read_raw_file(lq_bytes, is_gt=False)
        except Exception as e:
            raise Exception(f"LQ RAW corrupted: {lq_path}, err={e}")

        # -------------------
        # 数据增强
        # -------------------
        if self.opt['phase'] == 'train':

            gt_size = self.opt['gt_size']

            # padding 与原逻辑一致
            img_gt, img_lq = padding(img_gt, img_lq, gt_size)

            # crop
            img_gt, img_lq = paired_random_crop(img_gt, img_lq, gt_size, scale,
                                                gt_path)

            # flip / rot
            if self.geometric_augs:
                img_gt, img_lq = augment([img_gt, img_lq],
                                         self.opt['use_flip'],
                                         self.opt['use_rot'])

        # -------------------
        # RAW → tensor
        # -------------------
        img_gt, img_lq = img2tensor([img_gt, img_lq], bgr2rgb=False, float32=True)

        return {
            'lq': img_lq,
            'gt': img_gt,
            'lq_path': lq_path,
            'gt_path': gt_path
        }

    def __len__(self):
        return len(self.paths)


