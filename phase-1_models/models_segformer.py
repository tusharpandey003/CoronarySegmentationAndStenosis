# models_segformer.py

import torch.nn as nn
import segmentation_models_pytorch as smp

class SegFormerUnetModel(nn.Module):
    """
    A wrapper for a U-Net model with a Mix Transformer (MiT) encoder.
    This is a powerful and efficient transformer-based architecture supported
    by your installed library version.
    """
    def __init__(self, num_classes=2):
        super().__init__()
        
        # We use a U-Net architecture but replace the encoder with a Mix Transformer.
        # This is a 'SegFormer'-style architecture. We'll use the 'mit_b1' variant.
        self.model = smp.Unet(
            encoder_name="mit_b1",        # [FIXED] Use a supported transformer encoder
            encoder_weights="imagenet",   # Load weights pre-trained on ImageNet
            in_channels=3,
            classes=num_classes
        )
        print("[*] Initialized U-Net with a pre-trained Mix Transformer (mit_b1) encoder.")

    def forward(self, x):
        return self.model(x)