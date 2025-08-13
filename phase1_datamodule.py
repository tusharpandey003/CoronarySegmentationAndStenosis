import os
import random
import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
import numpy as np
import nibabel as nib
import cv2
from PIL import Image
# --- FIX APPLIED HERE: Using the older, compatible transforms API ---
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode

# ===================================================================
# 1. STANDARDIZED TRANSFORMS (Legacy API)
# ===================================================================
def get_image_transforms(is_train: bool, target_size: int = 512):
    """
    Returns a composed transform pipeline for images using the legacy API.
    """
    # Augmentations are applied to PIL Images
    augmentations = [
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(degrees=15),
    ] if is_train else []
    
    # Base transforms to resize, convert to tensor, and handle channels
    base_transforms = [
        T.ToPILImage(), # Convert NumPy array to PIL Image for augmentations
        T.Resize((target_size, target_size)),
        *augmentations, # Unpack augmentations here
        T.ToTensor(),   # Converts PIL to CHW Tensor and scales to [0.0, 1.0]
        T.Lambda(lambda x: x.repeat(3, 1, 1) if x.shape[0] == 1 else x), # Ensure 3 channels
    ]
    return T.Compose(base_transforms)

def get_mask_transforms(target_size: int = 512):
    """
    Returns a composed transform pipeline for masks using the legacy API.
    """
    return T.Compose([
        T.ToPILImage(),
        # Use NEAREST neighbor interpolation for masks to preserve class labels
        T.Resize((target_size, target_size), interpolation=InterpolationMode.NEAREST),
        # Convert the PIL image back to a NumPy array and then to a LongTensor
        T.Lambda(lambda img: torch.from_numpy(np.array(img, dtype=np.int64))),
    ])

# ===================================================================
# 2. INDIVIDUAL DATASET CLASSES (Updated for legacy transforms)
# ===================================================================
class ImageCASDataset(Dataset):
    """Dataset for IMAGE-CAS with specific windowing/normalization."""
    def __init__(self, file_list: list[tuple[str, str, int]], is_train: bool):
        super().__init__()
        self.slices = file_list
        self.transforms = get_image_transforms(is_train)
        self.mask_transforms = get_mask_transforms()
        self.window_min = -200
        self.window_max = 800
        print(f"[*] Initializing ImageCASDataset with {len(self.slices)} slices.")

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, idx):
        img_path, label_path, slice_idx = self.slices[idx]
        
        img_vol_data = nib.load(img_path).get_fdata()
        img_slice = img_vol_data[..., slice_idx].astype(np.float32)
        
        # Window and normalize the image to [0, 1] range
        img_slice = np.clip(img_slice, self.window_min, self.window_max)
        img_slice = (img_slice - self.window_min) / (self.window_max - self.window_min)
        # Convert to uint8 for PIL compatibility
        img_slice = (img_slice * 255).astype(np.uint8)
        
        mask_vol_data = nib.load(label_path).get_fdata()
        mask_slice = mask_vol_data[..., slice_idx].astype(np.uint8)
        mask_slice[mask_slice > 0] = 1
        
        return self.transforms(img_slice), self.mask_transforms(mask_slice)

class ArcadeDataset(Dataset):
    """Dataset for Arcade, updated for legacy transforms."""
    def __init__(self, images_dir: str, masks_dir: str, is_train: bool):
        super().__init__()
        self.image_paths = sorted([os.path.join(images_dir, f) for f in os.listdir(images_dir) if f.endswith('.png')])
        self.mask_paths = [os.path.join(masks_dir, os.path.basename(p)) for p in self.image_paths]
        self.transforms = get_image_transforms(is_train)
        self.mask_transforms = get_mask_transforms()
        print(f"[*] Initializing ArcadeDataset for dir: {images_dir} -> Found {len(self.image_paths)} pairs.")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image as NumPy array
        img = np.array(Image.open(self.image_paths[idx]))
        
        # Load mask and ensure it's binary
        mask = np.array(Image.open(self.mask_paths[idx])).astype(np.uint8)
        mask[mask > 0] = 1
        
        return self.transforms(img), self.mask_transforms(mask)

class DCA1Dataset(Dataset):
    """Dataset for DCA-1, updated for legacy transforms."""
    def __init__(self, file_pairs: list[dict], is_train: bool):
        super().__init__()
        self.file_pairs = file_pairs
        self.transforms = get_image_transforms(is_train)
        self.mask_transforms = get_mask_transforms()
        print(f"[*] Initializing DCA1Dataset with {len(self.file_pairs)} image pairs.")

    def __len__(self):
        return len(self.file_pairs)

    def __getitem__(self, idx):
        img_path = self.file_pairs[idx]['image']
        mask_path = self.file_pairs[idx]['mask']
        
        # Load image as NumPy array
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        
        # Load mask and ensure it's binary
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE).astype(np.uint8)
        mask[mask > 0] = 1
        
        return self.transforms(img), self.mask_transforms(mask)

# ===================================================================
# 3. MULTISOURCE WRAPPER, COLLATE, AND DATAMODULE
# ===================================================================
class MultiSourceDataset(Dataset):
    """A wrapper dataset that combines multiple individual datasets."""
    def __init__(self, datasets: list[Dataset], p_datasets: list[float] = None):
        self.datasets = datasets
        if p_datasets is None:
            self.p_datasets = [1.0 / len(datasets)] * len(datasets)
        else:
            self.p_datasets = p_datasets
        self.total_len = sum(len(ds) for ds in self.datasets)
        print(f"[*] Initializing MultiSourceDataset with {len(datasets)} datasets. Total length: {self.total_len}")

    def __len__(self):
        return self.total_len

    def __getitem__(self, idx):
        dataset_to_sample = random.choices(self.datasets, weights=self.p_datasets, k=1)[0]
        if len(dataset_to_sample) == 0:
            return None # Should be handled in collate
        random_index = random.randint(0, len(dataset_to_sample) - 1)
        return dataset_to_sample[random_index]

def segmentation_collate(batch):
    """Custom collate function that filters out None values."""
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    images, masks = zip(*batch)
    return torch.stack(images), torch.stack(masks)

class CoronaryDataModule(pl.LightningDataModule):
    """DataModule with manual 70/10/20 data splitting logic."""
    def __init__(self, data_config: dict, batch_size: int = 8, num_workers: int = 4):
        super().__init__()
        self.save_hyperparameters()
        self.data_config = data_config

    def setup(self, stage: str = None):
        print("-" * 50)
        print(f"Setting up data for stage: {stage}")

        # --- DCA-1 Data Splitting (70/10/20) ---
        all_dca1_files = []
        dca1_root = self.data_config['dca1_root']
        img_files = [f for f in os.listdir(dca1_root) if f.endswith('.pgm') and not f.endswith('_gt.pgm')]
        for img_file in img_files:
            mask_file = img_file.replace('.pgm', '_gt.pgm')
            if os.path.exists(os.path.join(dca1_root, mask_file)):
                all_dca1_files.append({"image": os.path.join(dca1_root, img_file), "mask": os.path.join(dca1_root, mask_file)})
        
        random.shuffle(all_dca1_files)
        train_end = int(0.7 * len(all_dca1_files))
        val_end = train_end + int(0.1 * len(all_dca1_files))
        dca1_train_files = all_dca1_files[:train_end]
        dca1_val_files = all_dca1_files[train_end:val_end]
        dca1_test_files = all_dca1_files[val_end:]

        # --- ImageCAS Data Splitting (70/10/20) ---
        all_cas_slices = []
        cas_root = self.data_config['imagecas_root']
        img_vols = sorted([f for f in os.listdir(cas_root) if f.endswith('.img.nii.gz')])
        for f in img_vols:
            pid = f.replace('.img.nii.gz', '')
            img_path = os.path.join(cas_root, f)
            label_path = os.path.join(cas_root, f"{pid}.label.nii.gz")
            if os.path.exists(label_path):
                label_vol = nib.load(label_path).get_fdata()
                for slice_idx in range(label_vol.shape[-1]):
                    if np.sum(label_vol[..., slice_idx]) >= 50:
                        all_cas_slices.append((img_path, label_path, slice_idx))
        
        random.shuffle(all_cas_slices)
        train_end = int(0.7 * len(all_cas_slices))
        val_end = train_end + int(0.1 * len(all_cas_slices))
        cas_train_slices = all_cas_slices[:train_end]
        cas_val_slices = all_cas_slices[train_end:val_end]
        cas_test_slices = all_cas_slices[val_end:]
        
        # --- Initialize Datasets for each split ---
        if stage == 'fit' or stage is None:
            self.train_dataset = MultiSourceDataset(datasets=[
                DCA1Dataset(dca1_train_files, is_train=True),
                ImageCASDataset(cas_train_slices, is_train=True),
                ArcadeDataset(self.data_config['arcade_syntax_train_images'], self.data_config['arcade_syntax_train_masks'], is_train=True)
            ])
            self.val_dataset = MultiSourceDataset(datasets=[
                DCA1Dataset(dca1_val_files, is_train=False),
                ImageCASDataset(cas_val_slices, is_train=False),
                ArcadeDataset(self.data_config['arcade_syntax_val_images'], self.data_config['arcade_syntax_val_masks'], is_train=False)
            ])
        if stage == 'test' or stage is None:
             self.test_dataset = MultiSourceDataset(datasets=[
                DCA1Dataset(dca1_test_files, is_train=False),
                ImageCASDataset(cas_test_slices, is_train=False),
                ArcadeDataset(self.data_config['arcade_syntax_test_images'], self.data_config['arcade_syntax_test_masks'], is_train=False)
            ])
        print("-" * 50)

    def train_dataloader(self): return DataLoader(self.train_dataset, batch_size=self.hparams.batch_size, shuffle=True, num_workers=self.hparams.num_workers, collate_fn=segmentation_collate, pin_memory=True)
    def val_dataloader(self): return DataLoader(self.val_dataset, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers, collate_fn=segmentation_collate, pin_memory=True)
    def test_dataloader(self): return DataLoader(self.test_dataset, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers, collate_fn=segmentation_collate, pin_memory=True)


