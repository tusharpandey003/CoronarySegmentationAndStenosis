import os, glob
import numpy as np
from PIL import Image
from collections import defaultdict
import torch
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.transforms as T

import pytorch_lightning as pl
from torch import nn
from torchvision import models
from sklearn.metrics import roc_auc_score, f1_score

# --- Severity mapping
SEVERITY_MAPS = {
    5: {'p0_20': 1, 'p0_20-50': 2, 'p0_50-70': 3, 'p0_70-90': 4, 'p0_90-98': 5},
    7: {'p0_20': 1, 'p0_20-50': 2, 'p0_50-70': 3, 'p0_70-90': 4, 'p0_90-98': 5, 'p0_99': 6, 'p0_100': 7}
}

# --- Dataset ---
class LesionDetectionDataset(Dataset):
    def __init__(self, root_dir, selected_folder='selectedVideos', transform=None, n_class=5):
        self.root_dir = root_dir
        self.selected_folder = selected_folder
        self.transform = transform
        self.n_class = n_class
        self.SEVMAP = SEVERITY_MAPS[n_class]
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
                    selected_frames = set(line.strip() for line in f if line.strip())
                for frame_id in selected_frames:
                    frame_path = os.path.join(input_dir, frame_id + ".png")
                    gt_file = os.path.join(gt_dir, f"{frame_id}.txt")
                    labels = []
                    if os.path.exists(gt_file):
                        with open(gt_file) as f2:
                            lines = [l.strip() for l in f2 if l.strip()]
                        for l in lines:
                            parts = l.split()
                            sev = parts[4]
                            if sev in self.SEVMAP:
                                labels.append(self.SEVMAP[sev])
                    binary_label = 1 if labels else 0
                    severity_label = max(labels) if labels else 0
                    self.samples.append({
                        'img': frame_path,
                        'binary_label': binary_label,
                        'severity_label': severity_label,
                        'frame_id': frame_id,
                    })
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img = Image.open(sample['img']).convert('RGB')
        if self.transform:
            img = self.transform(img)
        else:
            img = np.array(img)
            img = torch.from_numpy(img).permute(2,0,1).float() / 255.0
        return img, torch.tensor(sample['binary_label'], dtype=torch.float32), torch.tensor(sample['severity_label'], dtype=torch.long)

# --- Torchvision transform (suitable for ResNet) ---
def get_resnet_transform():
    return T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

# --- Split videos ---
def split_by_video(dataset, train_ratio=0.7, val_ratio=0.1, test_ratio=0.2, seed=42):
    video_to_indices = defaultdict(list)
    for idx, sample in enumerate(dataset.samples):
        video_id = '_'.join(sample['frame_id'].split('_')[:2])
        video_to_indices[video_id].append(idx)
    video_ids = list(video_to_indices.keys())
    np.random.seed(seed)
    np.random.shuffle(video_ids)
    n = len(video_ids)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    train_videos = video_ids[:n_train]
    val_videos = video_ids[n_train:n_train + n_val]
    test_videos = video_ids[n_train + n_val:]
    train_indices = [idx for vid in train_videos for idx in video_to_indices[vid]]
    val_indices = [idx for vid in val_videos for idx in video_to_indices[vid]]
    test_indices = [idx for vid in test_videos for idx in video_to_indices[vid]]
    return train_indices, val_indices, test_indices

# --- PyTorch Lightning DataModule ---
class LesionDataModule(pl.LightningDataModule):
    def __init__(self, root_dir, n_class=5, batch_size=32, num_workers=4):
        super().__init__()
        self.root_dir = root_dir
        self.n_class = n_class
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage=None):
        fullds = LesionDetectionDataset(
            self.root_dir, n_class=self.n_class, transform=get_resnet_transform()
        )
        train_idx, val_idx, test_idx = split_by_video(fullds)
        self.train_ds = Subset(fullds, train_idx)
        self.val_ds = Subset(fullds, val_idx)
        self.test_ds = Subset(fullds, test_idx)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True,
                          num_workers=self.num_workers, persistent_workers=(self.num_workers>0))

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, persistent_workers=(self.num_workers>0))

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, persistent_workers=(self.num_workers>0))

# --- LightningModule: ResNet-34 backbone ---
class LesionClassifier(pl.LightningModule):
    def __init__(self, backbone='resnet34', lr=1e-3):
        super().__init__()
        self.save_hyperparameters()
        if backbone == 'resnet34':
            self.model = models.resnet34(pretrained=True)
            self.model.fc = nn.Linear(self.model.fc.in_features, 1)
        else:
            raise ValueError("Only resnet34 is supported in this script for CUDA 10.2.")
        self.loss = nn.BCEWithLogitsLoss()
        self.lr = lr

    def forward(self, x):
        return self.model(x).view(-1)

    def training_step(self, batch, batch_idx):
        x, y, _ = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, severity = batch
        y_hat = self(x)
        return {'y_true': y.detach().cpu(), 'y_pred': y_hat.detach().cpu()}

    def test_step(self, batch, batch_idx):
        x, y, severity = batch
        y_hat = self(x)
        return {'y_true': y.detach().cpu(), 'y_pred': y_hat.detach().cpu(), 'severity': severity.detach().cpu()}

    def validation_epoch_end(self, outputs):
        y_true = torch.cat([x['y_true'] for x in outputs])
        y_pred = torch.cat([x['y_pred'] for x in outputs])
        pred_prob = torch.sigmoid(y_pred.float()).cpu()
        y_true = y_true.cpu()
        pred_bin = (pred_prob > 0.5).float()
        auc = roc_auc_score(y_true, pred_prob)
        f1 = f1_score(y_true, pred_bin)
        self.log('val_auc', auc, prog_bar=True)
        self.log('val_f1', f1, prog_bar=True)
        if hasattr(self, "trainer") and self.current_epoch == self.trainer.max_epochs - 1:
            print(f"Final Validation Results - Epoch {self.current_epoch}: AUC={auc:.4f}, F1={f1:.4f}")

    def test_epoch_end(self, outputs):
        y_true = torch.cat([x['y_true'] for x in outputs])
        y_pred = torch.cat([x['y_pred'] for x in outputs])
        pred_prob = torch.sigmoid(y_pred.float()).cpu()
        y_true = y_true.cpu()
        pred_bin = (pred_prob > 0.5).float()
        auc = roc_auc_score(y_true, pred_prob)
        f1 = f1_score(y_true, pred_bin)
        self.log('test_auc', auc)
        self.log('test_f1', f1)
        print(f"Test Results: AUC={auc:.4f}, F1={f1:.4f}")

        severity = torch.cat([x['severity'] for x in outputs])
        severity = severity.cpu()
        for sev in torch.unique(severity):
            idx = severity == sev
            labels_subset = y_true[idx]
            if len(torch.unique(labels_subset)) < 2:
                print(f"Skipping severity {sev.item()} - only one class present")
                continue
            auc_sev = roc_auc_score(labels_subset, pred_prob[idx])
            f1_sev = f1_score(labels_subset, pred_bin[idx])
            self.log(f'test_auc_sev{sev.item()}', auc_sev)
            self.log(f'test_f1_sev{sev.item()}', f1_sev)
            print(f"Test Stratified Severity {sev.item()}: AUC={auc_sev:.4f}, F1={f1_sev:.4f}")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

# --- Main experiment loop ---
def run_experiments(data_path, experiments, max_epochs=50, batch_size=256, lr=1e-3, device='gpu', num_workers=4):
    results = []
    for n_class, backbone in experiments:
        print(f"\n=== Starting experiment: n_class={n_class}, backbone={backbone} ===\n")
        dm = LesionDataModule(
            root_dir=data_path, n_class=n_class, batch_size=batch_size, num_workers=num_workers
        )
        model = LesionClassifier(backbone=backbone, lr=lr)
        trainer = pl.Trainer(
            max_epochs=max_epochs,
            devices=[1,2],           # change list if you have fewer GPUs, or use devices=1 for single GPU
            accelerator='gpu',
            strategy='ddp',
            precision=16,              # Only if supported by your CUDA hardware and driver
            logger=pl.loggers.CSVLogger("logs"),
            log_every_n_steps=10
        )
        trainer.fit(model, datamodule=dm)
        test_results = trainer.test(model, datamodule=dm)
        print(f"Experiment finished: n_class={n_class}, backbone={backbone}")
        print(f"Test Results: {test_results}")
        results.append({
            'n_class': n_class,
            'backbone': backbone,
            'test_metrics': test_results
        })
    return results

if __name__ == '__main__':
    DATA_PATH = "/home/tushar/cadica/CADICA" # Adjust the path as needed

    experiment_list = [
        (5, 'resnet34'),
        (7, 'resnet34'),
        
    ]
    all_results = run_experiments(DATA_PATH, experiment_list)

    print("\nSummary of All Experiments:")
    for res in all_results:
        print(f"n_class={res['n_class']}, backbone={res['backbone']}, test_metrics={res['test_metrics']}")
