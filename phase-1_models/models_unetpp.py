# models_unetpp.py

import torch.nn as nn
import segmentation_models_pytorch as smp

class ArterySegmentationModel(nn.Module):
    """
    A wrapper for the UNet++ model from the segmentation-models-pytorch library.
    This model is significantly lighter and well-suited for a 12GB VRAM environment.
    """
    def __init__(self, num_classes=2):
        super().__init__()
        
        # We use UNet++ with a ResNet34 backbone pre-trained on ImageNet.
        # This provides a great balance of performance and resource efficiency.
        # The library handles creating the encoder, decoder, and skip connections.
        self.model = smp.UnetPlusPlus(
            encoder_name="resnet34",      # A light and effective backbone
            encoder_weights="imagenet",   # Leverage transfer learning
            in_channels=3,                # Your dataloader provides 3-channel images
            classes=num_classes           # Output channels (background, artery)
        )
        print("[*] Initialized UNet++ model with a pre-trained ResNet34 encoder.")

    def forward(self, x):
        return self.model(x)