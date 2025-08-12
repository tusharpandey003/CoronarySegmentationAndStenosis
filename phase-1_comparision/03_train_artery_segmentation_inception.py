# 03_train_artery_segmentation_inception.py

import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import argparse

from segmentation_models_pytorch.losses import DiceLoss
from phase1_datamodule import CoronaryDataModule
# [MODIFIED] Import the new Inception-based model
from models_inception import InceptionUnetModel

class ArterySegmentationTask(pl.LightningModule):
    # This class remains unchanged as it is model-agnostic.
    def __init__(self, model, learning_rate=1e-4):
        super().__init__()
        self.save_hyperparameters(ignore=['model'])
        self.model = model
        self.loss_fn = DiceLoss(mode='multiclass', from_logits=True)

    def forward(self, x):
        return self.model(x)

    def _common_step(self, batch, batch_idx):
        images, masks = batch
        masks = masks.long()
        logits = self.forward(images)
        loss = self.loss_fn(logits, masks)
        preds = torch.argmax(logits, dim=1)
        
        tp = torch.sum((preds == 1) & (masks == 1)).float()
        tn = torch.sum((preds == 0) & (masks == 0)).float()
        fp = torch.sum((preds == 1) & (masks == 0)).float()
        fn = torch.sum((preds == 0) & (masks == 1)).float()
        epsilon = 1e-6

        sensitivity = tp / (tp + fn + epsilon)
        specificity = tn / (tn + fp + epsilon)
        accuracy = (tp + tn) / (tp + tn + fp + fn + epsilon)
        dice_score = (2 * tp) / (2 * tp + fp + fn + epsilon)
        iou = tp / (tp + fp + fn + epsilon)
        
        metrics = {
            "loss": loss, "dice_score": dice_score, "iou": iou,
            "sensitivity": sensitivity, "specificity": specificity, "accuracy": accuracy
        }
        return metrics

    def training_step(self, batch, batch_idx):
        metrics = self._common_step(batch, batch_idx)
        self.log_dict({f"train_{k}": v for k, v in metrics.items()}, on_step=False, on_epoch=True, prog_bar=True)
        return metrics["loss"]

    def validation_step(self, batch, batch_idx):
        metrics = self._common_step(batch, batch_idx)
        self.log_dict({f"val_{k}": v for k, v in metrics.items()}, on_epoch=True, prog_bar=True)
        
    def test_step(self, batch, batch_idx):
        metrics = self._common_step(batch, batch_idx)
        self.log_dict({f"test_{k}": v for k, v in metrics.items()}, on_epoch=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5, factor=0.5)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_dice_score"},
        }

def main(args):
    pl.seed_everything(42, workers=True)

    data_paths = {
        'arcade_syntax_train_images': args.arcade_train_img, 'arcade_syntax_train_masks': args.arcade_train_mask,
        'arcade_syntax_val_images': args.arcade_val_img, 'arcade_syntax_val_masks': args.arcade_val_mask,
        'arcade_syntax_test_images': args.arcade_test_img, 'arcade_syntax_test_masks': args.arcade_test_mask,
        'dca1_root': args.dca1_root,
        'imagecas_root': args.imagecas_root,
    }
    datamodule = CoronaryDataModule(data_config=data_paths, batch_size=args.batch_size, num_workers=args.num_workers)
    
    # [MODIFIED] Instantiate the new InceptionUnetModel
    model = InceptionUnetModel(num_classes=2)
    task = ArterySegmentationTask(model, learning_rate=args.lr)

    # [MODIFIED] Changed paths for this specific experiment
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath='checkpoints_inception',
        filename='artery-seg-inception-{epoch:02d}-{val_dice_score:.4f}',
        save_top_k=1,
        verbose=True,
        monitor='val_dice_score',
        mode='max'
    )
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='epoch')

    # [MODIFIED] Changed logger name for separate tracking in TensorBoard
    trainer = pl.Trainer(
        gpus=args.gpus,
        max_epochs=args.epochs,
        callbacks=[checkpoint_callback, lr_monitor],
        logger=pl.loggers.TensorBoardLogger('logs/', name='artery_segmentation_inception'),
        precision=16 if args.use_amp else 32,
        log_every_n_steps=10
    )

    trainer.fit(task, datamodule)
    
    print("\n" + "="*50)
    print("      STARTING FINAL EVALUATION ON TEST SET (Inception-Unet)")
    print("="*50)
    
    test_results = trainer.test(datamodule=datamodule, ckpt_path='best')
    
    print("\n" + "="*50)
    print("             TESTING COMPLETE (Inception-Unet)")
    print("="*50)
    print("Final Test Metrics:")
    if test_results:
        final_metrics = test_results[0]
        for key, value in final_metrics.items():
            print(f"  {key.replace('test_', '').capitalize()}: {value:.4f}")
    print("="*50 + "\n")

    best_model_path = checkpoint_callback.best_model_path
    # [MODIFIED] Save a uniquely named backbone file
    final_checkpoint_path = 'artery_backbone_inception.pth'
    
    best_task = ArterySegmentationTask.load_from_checkpoint(best_model_path)
    torch.save(best_task.model.state_dict(), final_checkpoint_path)
    print(f"✅ Training complete. Best Inception-Unet checkpoint at: {best_model_path}")
    print(f"✅ Final Inception-Unet backbone for Phase 2 saved to: {final_checkpoint_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train Phase 1 Artery Segmentation Model using InceptionResNetV2-U-Net")
    parser.add_argument('--arcade_train_img', type=str, required=True)
    parser.add_argument('--arcade_train_mask', type=str, required=True)
    parser.add_argument('--arcade_val_img', type=str, required=True)
    parser.add_argument('--arcade_val_mask', type=str, required=True)
    parser.add_argument('--arcade_test_img', type=str, required=True)
    parser.add_argument('--arcade_test_mask', type=str, required=True)
    parser.add_argument('--dca1_root', type=str, required=True)
    parser.add_argument('--imagecas_root', type=str, required=True)
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=50)
    # [MODIFIED] Reduced default batch size for this heavy model
    parser.add_argument('--batch_size', type=int, default=4, help='InceptionResNetV2 is heavy; a small batch size is recommended')
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--use_amp', action='store_true', help='Use Automatic Mixed Precision (precision=16)')
    
    args = parser.parse_args()
    main(args)