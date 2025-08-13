# models_densenet.py

import torch.nn as nn
import segmentation_models_pytorch as smp

class DenseNetUnetModel(nn.Module):
    """
    A wrapper for a U-Net model with a DenseNet backbone.
    DenseNet's core idea is "feature reuse," where each layer receives feature
    maps from all preceding layers, encouraging the learning of compact and
    robust features.
    """
    def __init__(self, num_classes=2):
        super().__init__()

        # Instantiate the U-Net model with the DenseNet121 encoder
        self.model = smp.Unet(
            encoder_name="densenet121",      # A parameter-efficient and powerful encoder
            encoder_weights="imagenet",       # Load pre-trained weights
            in_channels=3,
            classes=num_classes
        )
        print("[*] Initialized U-Net with a pre-trained DenseNet121 encoder.")

    def forward(self, x):
        return self.model(x)