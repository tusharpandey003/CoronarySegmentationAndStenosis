import os
import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from torchvision.models import swin_t, Swin_T_Weights
import torchvision.transforms as transforms
from monai.losses.dice import DiceLoss
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from segmentation_models_pytorch.losses import DiceLoss  # [3][4]
from catalyst.metrics import dice as catalyst_dice


# 1. Define normalization transform for 1-channel input
IMG_MEAN = [0.1238]
IMG_STD = [0.1606]
my_transform = transforms.Compose([
    transforms.ToTensor(),  # expects (H, W) -> (1, H, W), scales to [0,1]
    transforms.Resize((256, 256)),
    transforms.Normalize(mean=IMG_MEAN, std=IMG_STD)
])

# 2. Dataset with transform


class NiftiPairDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.slices = []
        for fname in os.listdir(data_dir):
            if fname.endswith('.img.nii.gz'):
                patient_id = fname.split('.')[0]
                img_path = os.path.join(data_dir, f"{patient_id}.img.nii.gz")
                label_path = os.path.join(
                    data_dir, f"{patient_id}.label.nii.gz")
                if os.path.exists(label_path):
                    img_vol = nib.load(img_path).get_fdata()
                    num_slices = img_vol.shape[-1]
                    for i in range(num_slices):
                        self.slices.append((img_path, label_path, i))

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, idx):
        img_path, label_path, slice_idx = self.slices[idx]
        img_vol = nib.load(img_path).get_fdata()
        label_vol = nib.load(label_path).get_fdata()
        img_slice = img_vol[..., slice_idx]
        label_slice = label_vol[..., slice_idx]

        # Min-max normalize to [0, 1] for image
        img_slice = (img_slice - np.min(img_slice)) / \
            (np.max(img_slice) - np.min(img_slice) + 1e-8)
        img_slice = img_slice.astype(np.float32)
        label_slice = label_slice.astype(np.int64)

        # Apply transform to image (includes ToTensor, Resize, Normalize)
        if self.transform:
            img_slice = self.transform(img_slice)
        else:
            img_slice = torch.tensor(
                img_slice, dtype=torch.float32).unsqueeze(0)

        # Label: to tensor, add channel, resize, then squeeze back to (H, W)
        label_slice = torch.tensor(label_slice, dtype=torch.long).unsqueeze(0)
        label_slice = F.interpolate(label_slice.unsqueeze(
            0).float(), size=(256, 256), mode='nearest').squeeze(0).long()
        return img_slice, label_slice

# 3. LightningDataModule


class NiftiDataModule(pl.LightningDataModule):
    def __init__(self, train_dir, val_dir, test_dir, batch_size=60, num_workers=12, transform=None):
        super().__init__()
        self.train_dir = train_dir
        self.val_dir = val_dir
        self.test_dir = test_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transform = transform

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_dataset = NiftiPairDataset(
                self.train_dir, transform=self.transform)
            self.val_dataset = NiftiPairDataset(
                self.val_dir, transform=self.transform)
        if stage == "test" or stage is None:
            self.test_dataset = NiftiPairDataset(
                self.test_dir, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True,
                          num_workers=self.num_workers, pin_memory=True, persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers, pin_memory=True, persistent_workers=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers, pin_memory=True, persistent_workers=True)

# -------- LightningModule --------


class SwinTSegmentationModel(pl.LightningModule):
    def __init__(self, in_channels=1, classes=18, lr=1e-4):
        super().__init__()
        self.save_hyperparameters()
        self.backbone = swin_t(weights=Swin_T_Weights.IMAGENET1K_V1)
        self.backbone.features[0][0] = nn.Conv2d(
            in_channels, 96, kernel_size=4, stride=4, bias=False)
        self.segmentation_head = nn.Conv2d(768, classes, kernel_size=1)
        self.lr = lr
        self.num_classes = classes
        # Use DiceLoss from SMP in multiclass mode, expects logits
        self.loss_fn = DiceLoss(mode='multiclass', from_logits=True)  # [3][4]

    def forward(self, x):
        features = self.backbone.features(x)
        features = features.permute(
            0, 3, 1, 2).contiguous()  # Add .contiguous() here
        out = self.segmentation_head(features)
        out = F.interpolate(out, size=(256, 256),
                            mode='bilinear', align_corners=False)
        return out

    def _shared_step(self, batch, stage="val"):
        images, masks = batch
        logits = self(images)
        preds = torch.argmax(logits, dim=1)
        # Dice loss (multiclass, from_logits)
        dice_loss = self.loss_fn(logits, masks.squeeze(1))
        # Pixel accuracy
        correct = (preds == masks.squeeze(1)).sum().item()
        total = preds.numel()
        pixel_acc = correct / total
        # IoU
        ious = []
        for c in range(self.num_classes):
            pred_c = (preds == c)
            label_c = (masks.squeeze(1) == c)
            intersection = (pred_c & label_c).sum().item()
            union = (pred_c | label_c).sum().item()
            if union > 0:
                ious.append(intersection / union)
        mean_iou = np.mean(ious) if ious else 0
        # Catalyst Dice (for per-class reporting, optional)
        # You may skip this if only using SMP DiceLoss for training
        preds_onehot = F.one_hot(
            preds, num_classes=self.num_classes).permute(0, 3, 1, 2).float()
        labels_onehot = F.one_hot(masks.squeeze(
            1), num_classes=self.num_classes).permute(0, 3, 1, 2).float()
        dice_scores = catalyst_dice(
            outputs=preds_onehot, targets=labels_onehot, class_dim=1, threshold=0.5)
        if dice_scores.ndim > 1:
            dice_scores = dice_scores.mean(dim=0)
        mean_dice = dice_scores.mean().item()
        # Logging
        self.log(f"{stage}_loss", dice_loss, on_step=False,
                 on_epoch=True, prog_bar=True)
        self.log(f"{stage}_pixel_acc", pixel_acc,
                 on_step=False, on_epoch=True, prog_bar=True)
        self.log(f"{stage}_iou", mean_iou, on_step=False,
                 on_epoch=True, prog_bar=True)
        self.log(f"{stage}_dice", mean_dice, on_step=False,
                 on_epoch=True, prog_bar=True)
        return {
            "loss": dice_loss,
            "pixel_acc": pixel_acc,
            "iou": mean_iou,
            "dice": mean_dice,
            "dice_per_class": dice_scores.cpu().numpy()
        }

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, stage="train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, stage="val")

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, stage="test")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    # def test_step(self, batch, batch_idx):
    #     images, seg_masks, cls_labels = batch
    #     seg_logits, cls_logit = self(images)
    #     seg_loss = self.dice_loss(seg_logits, F.one_hot(seg_masks.squeeze(
    #         1), num_classes=self.hparams.seg_classes).permute(0, 3, 1, 2).float())
    #     cls_loss = self.cls_loss_fn(cls_logit.squeeze(1), cls_labels.float())
    #     dice_score = dice_score(seg_logits, )
    #     auroc = AUROC(images, true_labels)
    #     loss = seg_loss + self.cls_weight * cls_loss
    #     self.log('val_loss', loss, on_epoch=True, prog_bar=True)
    #     self.log('val_IOU', loss, on_epoch=True, prog_bar=True)
    #     self.log('val_DICE', loss, on_epoch=True, prog_bar=True)
    #     self.log('val_auroc', loss, on_epoch=True, prog_bar=True)
    #     # self.log('val_dice', 1-seg_loss, on_epoch=True, prog_bar=True)
    #     # self.log('val_cls_loss', cls_loss, on_epoch=True, prog_bar=True)
    #     return {'val_loss': loss, 'val_dice': 1-seg_loss, 'val_cls_loss': cls_loss}


# 4. Training Script with ModelCheckpoint callback

if __name__ == "__main__":
    train_dir = '/home/tushar/train'
    val_dir = '/home/tushar/val'
    test_dir = '/home/tushar/test'
    batch_size = 60
    num_workers = 12

    # DataLoaders
    train_dataset = NiftiPairDataset(train_dir, transform=my_transform)
    val_dataset = NiftiPairDataset(val_dir, transform=my_transform)
    test_dataset = NiftiPairDataset(test_dir, transform=my_transform)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    model = SwinTSegmentationModel(
        in_channels=1, classes=18, lr=1e-4)

    # ModelCheckpoint callback
    checkpoint_callback = ModelCheckpoint(
        monitor="val_dice",
        save_top_k=2,
        mode="max",
        dirpath="./checkpoints",
        filename="swinT-{epoch:02d}-{val_dice:.4f}"
    )

    # EarlyStopping callback
    early_stopping_callback = EarlyStopping(
        monitor="val_dice",
        patience=2,
        mode="max"
    )

    trainer = pl.Trainer(
        max_epochs=20,
        accelerator="gpu",
        log_every_n_steps=10,
        strategy='ddp',
        sync_batchnorm=True,
        devices=[1, 2],
        precision=16,
        callbacks=[checkpoint_callback, early_stopping_callback]
    )
    # Training and validation
    trainer.fit(model, train_dataloaders=train_loader,
                val_dataloaders=val_loader)

    # Testing (automatically runs test_step)
    test_results = trainer.test(model, dataloaders=test_loader)
    print("Test Results:", test_results)
