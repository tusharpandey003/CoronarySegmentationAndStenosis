import os
import glob
import random
import numpy as np
from PIL import Image
from collections import defaultdict
import torch
from torch.utils.data import Dataset, DataLoader, Subset
import pytorch_lightning as pl
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

# ---- Severity map ----
SEVERITY_MAP = {
    'p0': 0,     # none
    'p0_20': 1,  # mild
    'p20_50': 2,  # moderate
    'p50_70': 3  # severe
}
CLASS_NAMES = ['none', 'mild', 'moderate', 'severe']

# ---- Dataset ----


class LesionSeverityDataset(Dataset):
    def __init__(self, root_dir, selected_folder='selectedVideos', transform=None):
        self.root_dir = root_dir
        self.selected_folder = selected_folder
        self.transform = transform
        self.samples = []
        self._prepare_samples()

    def _prepare_samples(self):
        base_dir = os.path.join(self.root_dir, self.selected_folder)
        for patient_dir in glob.glob(os.path.join(base_dir, 'p*')):
            for video_dir in glob.glob(os.path.join(patient_dir, 'v*')):
                input_dir = os.path.join(video_dir, 'input')
                gt_dir = os.path.join(video_dir, 'groundtruth')
                selected_frames_file = os.path.join(
                    video_dir,
                    f"{os.path.basename(patient_dir)}_{os.path.basename(video_dir)}_selectedFrames.txt"
                )
                if not os.path.exists(selected_frames_file):
                    continue
                with open(selected_frames_file) as f:
                    selected_frames = set(line.strip()
                                          for line in f if line.strip())
                for frame_id in selected_frames:
                    frame_path = os.path.join(input_dir, frame_id + ".png")
                    gt_file = os.path.join(gt_dir, f"{frame_id}.txt")
                    if not os.path.exists(frame_path):
                        continue
                    if os.path.exists(gt_file):
                        with open(gt_file) as f:
                            lines = [l.strip() for l in f if l.strip()]
                        boxes, labels = [], []
                        for l in lines:
                            parts = l.split()
                            x, y, w, h = map(int, parts[:4])
                            sev = parts[4]
                            if sev not in SEVERITY_MAP:
                                continue
                            boxes.append([x, y, x + w, y + h])
                            labels.append(SEVERITY_MAP[sev])
                        if boxes:
                            self.samples.append({
                                'img': frame_path,
                                'boxes': np.array(boxes, dtype=np.float32),
                                'labels': np.array(labels, dtype=np.int64),
                                'frame_id': frame_id
                            })
                        else:
                            self.samples.append({
                                'img': frame_path,
                                'boxes': np.array([[0, 0, 1, 1]], dtype=np.float32),
                                'labels': np.array([0], dtype=np.int64),
                                'frame_id': frame_id
                            })
                    else:
                        self.samples.append({
                            'img': frame_path,
                            'boxes': np.array([[0, 0, 1, 1]], dtype=np.float32),
                            'labels': np.array([0], dtype=np.int64),
                            'frame_id': frame_id
                        })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img = np.array(Image.open(sample['img']).convert('RGB'))
        if self.transform:
            img = self.transform(img)
        else:
            img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        target = {
            'boxes': torch.as_tensor(sample['boxes'], dtype=torch.float32),
            'labels': torch.as_tensor(sample['labels'], dtype=torch.int64),
            'image_id': torch.tensor([idx])
        }
        return img, target

# ---- Split by video ----


def split_by_video(dataset, train_ratio=0.7, val_ratio=0.1, test_ratio=0.2, seed=42):
    video_to_indices = defaultdict(list)
    for idx, sample in enumerate(dataset.samples):
        video_id = '_'.join(sample['frame_id'].split('_')[:2])
        video_to_indices[video_id].append(idx)
    video_ids = list(video_to_indices.keys())
    random.seed(seed)
    random.shuffle(video_ids)
    n = len(video_ids)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train_videos = video_ids[:n_train]
    val_videos = video_ids[n_train:n_train+n_val]
    test_videos = video_ids[n_train+n_val:]
    train_indices = [
        idx for vid in train_videos for idx in video_to_indices[vid]]
    val_indices = [idx for vid in val_videos for idx in video_to_indices[vid]]
    test_indices = [
        idx for vid in test_videos for idx in video_to_indices[vid]]
    return train_indices, val_indices, test_indices

# ---- Collate function ----


def collate_fn(batch):
    return tuple(zip(*batch))

# ---- DataModule ----


class LesionDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size=8):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size

    def setup(self, stage=None):
        self.dataset = LesionSeverityDataset(self.data_dir)
        self.train_idx, self.val_idx, self.test_idx = split_by_video(
            self.dataset)
        self.train_set = Subset(self.dataset, self.train_idx)
        self.val_set = Subset(self.dataset, self.val_idx)
        self.test_set = Subset(self.dataset, self.test_idx)

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True, collate_fn=collate_fn)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, shuffle=False, collate_fn=collate_fn)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False, collate_fn=collate_fn)

# ---- LightningModule for Faster R-CNN ----


class FasterRCNNLitModule(pl.LightningModule):
    def __init__(self, num_classes=4, lr=1e-4):
        super().__init__()
        self.save_hyperparameters()
        self.model = fasterrcnn_resnet50_fpn(weights="DEFAULT")
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(
            in_features, num_classes)
        self.lr = lr

    def forward(self, images, targets=None):
        if targets is not None:
            return self.model(images, targets)
        else:
            return self.model(images)

    def training_step(self, batch, batch_idx):
        images, targets = batch
        loss_dict = self(images, targets)
        loss = sum(loss for loss in loss_dict.values())
        self.log("train_loss", loss, on_step=True,
                 on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        # Ensure model is in train mode for loss calculation
        self.model.train()
        loss_dict = self.model(images, targets)
        loss = sum(loss for loss in loss_dict.values())
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return {"val_loss": loss}

    def test_step(self, batch, batch_idx):
        images, targets = batch
        # Move images and targets to the correct device if needed
        images = [img.to(self.device) for img in images]
        for t in targets:
            t['boxes'] = t['boxes'].to(self.device)
            t['labels'] = t['labels'].to(self.device)
        outputs = self(images)
        return {"outputs": outputs, "targets": targets}

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        images, targets = batch
        return {"outputs": self(images), "targets": targets}

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)


# ---- Main script ----
if __name__ == "__main__":
    data_dir = "/home/tushar/cadica/CADICA"
    batch_size = 10
    num_classes = 4

    # 1. Setup data module
    dm = LesionDataModule(data_dir, batch_size=batch_size)
    dm.setup()
    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()
    test_loader = dm.test_dataloader()

    # 2. Initialize LightningModule
    model = FasterRCNNLitModule(num_classes=num_classes, lr=1e-4)

    # 3. Callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        dirpath='checkpoints_new/',
        filename='best-checkpoint',
        save_top_k=1,
        mode='min'
    )
    progress_bar = TQDMProgressBar(refresh_rate=20)

    # 4. Trainer setup
    trainer = pl.Trainer(
        max_epochs=1,
        accelerator="gpu",
        devices=[1, 2],  # set to 1 for single GPU, or [0,1,...] for multi-GPU
        strategy="ddp",
        sync_batchnorm=True,
        precision=16,
        log_every_n_steps=10,
        callbacks=[checkpoint_callback, progress_bar],
        gradient_clip_val=1.0
    )

    # 5. Train and validate
    trainer.fit(model, train_dataloaders=train_loader,
                val_dataloaders=val_loader)

    # 6. Test
    trainer.test(model, dataloaders=test_loader)

    # 7. Predict and evaluate
    predictions = trainer.predict(model, dataloaders=test_loader)

    # --- Object Detection Metrics (mAP, mAR, etc.) ---
    map_metric = MeanAveragePrecision(box_format='xyxy', iou_type='bbox')
    for batch in predictions:
        outputs = batch["outputs"]
        targets = batch["targets"]
        map_metric.update(outputs, targets)
    results = map_metric.compute()
    print(f"Mean Average Precision (mAP): {results['map'].item():.4f}")
    print(f"Mean Average Recall (mAR): {results['mar_100'].item():.4f}")
    print(f"mAP@0.5: {results['map_50'].item():.4f}")
    print(f"mAP@0.75: {results['map_75'].item():.4f}")
    print("Per-class AP:", results["map_per_class"])

    # --- Top-1 Classification Metrics (Accuracy, F1, Confusion Matrix) ---
    all_preds = []
    all_gts = []
    for batch in predictions:
        outputs = batch["outputs"]
        targets = batch["targets"]
        for pred, gt in zip(outputs, targets):
            if len(pred['labels']) > 0:
                all_preds.append(pred['labels'][0].cpu().item())
            else:
                all_preds.append(0)
            if len(gt['labels']) > 0:
                all_gts.append(gt['labels'][0].cpu().item())
            else:
                all_gts.append(0)
    acc = accuracy_score(all_gts, all_preds)
    f1 = f1_score(all_gts, all_preds, average='weighted')
    cm = confusion_matrix(all_gts, all_preds)
    print(f"Test Accuracy: {acc:.4f} | F1: {f1:.4f}")
    print("Confusion Matrix:\n", cm)
