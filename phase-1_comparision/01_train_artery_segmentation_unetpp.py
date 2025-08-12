# 01_train_artery_segmentation_unetpp.py (Corrected for your environment)

import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import argparse

# --- Library Imports ---
# [FIXED] Use the DiceLoss from segmentation-models-pytorch as you suggested.
from segmentation_models_pytorch.losses import DiceLoss

# --- Local Imports ---
from phase1_datamodule import CoronaryDataModule
from models_unetpp import ArterySegmentationModel

class ArterySegmentationTask(pl.LightningModule):
    def __init__(self, model, learning_rate=1e-4):
        super().__init__()
        self.save_hyperparameters(ignore=['model'])
        self.model = model
        
        # [MODIFIED] Using the DiceLoss from SMP, which is stable in your environment.
        
        self.loss_fn = DiceLoss(mode='multiclass', from_logits=True)

    def forward(self, x):
        return self.model(x)

    def _common_step(self, batch, batch_idx):
        """
        This function is now the core for both loss and all manual metric calculations.
        """
        images, masks = batch
        masks = masks.long() # Ensure masks are long type for calculations

        # Forward pass
        logits = self.forward(images)

        # 1. Calculate Loss
        loss = self.loss_fn(logits, masks)

        # 2. Calculate Predictions for Metrics
        preds = torch.argmax(logits, dim=1)

        # 3. Manually Calculate Metrics for the foreground class (class 1)
        # True Positives (TP): correctly predicted artery pixels
        # True Negatives (TN): correctly predicted background pixels
        # False Positives (FP): background predicted as artery
        # False Negatives (FN): artery predicted as background
        
        tp = torch.sum((preds == 1) & (masks == 1)).float()
        tn = torch.sum((preds == 0) & (masks == 0)).float()
        fp = torch.sum((preds == 1) & (masks == 0)).float()
        fn = torch.sum((preds == 0) & (masks == 1)).float()

        # Handle division by zero for stability
        epsilon = 1e-6

        # Sensitivity (Recall) = TP / (TP + FN)
        sensitivity = tp / (tp + fn + epsilon)
        
        # Specificity = TN / (TN + FP)
        specificity = tn / (tn + fp + epsilon)
        
        # Accuracy = (TP + TN) / (TP + TN + FP + FN)
        accuracy = (tp + tn) / (tp + tn + fp + fn + epsilon)
        
        # Dice Score = 2*TP / (2*TP + FP + FN)
        dice_score = (2 * tp) / (2 * tp + fp + fn + epsilon)

        # IoU (Jaccard) = TP / (TP + FP + FN)
        iou = tp / (tp + fp + fn + epsilon)
        
        # Return a dictionary of all computed values
        metrics = {
            "loss": loss,
            "dice_score": dice_score,
            "iou": iou,
            "sensitivity": sensitivity,
            "specificity": specificity,
            "accuracy": accuracy
        }
        return metrics

    def training_step(self, batch, batch_idx):
        metrics = self._common_step(batch, batch_idx)
        # Use log_dict for cleaner logging
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
        # Monitor validation dice score, as it's a primary segmentation metric
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
    
    model = ArterySegmentationModel(num_classes=2)
    task = ArterySegmentationTask(model, learning_rate=args.lr)

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath='checkpoints_unetpp',
        filename='artery-seg-unetpp-{epoch:02d}-{val_dice_score:.4f}',
        save_top_k=1,
        verbose=True,
        monitor='val_dice_score',
        mode='max'
    )
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='epoch')

    trainer = pl.Trainer(accelerator='gpu',
        devices=[0],
        max_epochs=args.epochs,
        callbacks=[checkpoint_callback, lr_monitor],
        logger=pl.loggers.TensorBoardLogger('logs/', name='artery_segmentation_unetpp'),
        precision=16 if args.use_amp else 32,
        log_every_n_steps=10
    )

    trainer.fit(task, datamodule)
    
    print("\n" + "="*50)
    print("      STARTING FINAL EVALUATION ON TEST SET")
    print("="*50)
    
    test_results = trainer.test(datamodule=datamodule, ckpt_path='best')
    
    print("\n" + "="*50)
    print("             TESTING COMPLETE")
    print("="*50)
    print("Final Test Metrics:")
    if test_results:
        # Results from trainer.test is a list of dicts
        final_metrics = test_results[0]
        for key, value in final_metrics.items():
            # The key will be like 'test_dice_score'
            print(f"  {key.replace('test_', '').capitalize()}: {value:.4f}")
    print("="*50 + "\n")


    best_model_path = checkpoint_callback.best_model_path
    final_checkpoint_path = 'artery_backbone.pth'
    
    best_task = ArterySegmentationTask.load_from_checkpoint(best_model_path)
    torch.save(best_task.model.state_dict(), final_checkpoint_path)
    print(f"✅ Training complete. Best model checkpoint at: {best_model_path}")
    print(f"✅ Final UNet++ backbone for Phase 2 saved to: {final_checkpoint_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train Phase 1 Artery Segmentation Model using UNet++")
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
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--use_amp', action='store_true', help='Use Automatic Mixed Precision (precision=16)')
    
    args = parser.parse_args()
    main(args)