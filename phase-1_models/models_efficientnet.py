# models_efficientnet.py

import torch.nn as nn
import segmentation_models_pytorch as smp

class EfficientNetUnetModel(nn.Module):
    """
    A wrapper for a U-Net model with an EfficientNet backbone.
    EfficientNet is known for its excellent performance-to-computation ratio,
    achieved through principled compound scaling of network depth, width, and resolution.
    """
    def __init__(self, num_classes=2):
        super().__init__()

        # Instantiate the U-Net model with the EfficientNet-B4 encoder
        self.model = smp.Unet(
            encoder_name="efficientnet-b4",  # A strong and efficient encoder
            encoder_weights="imagenet",       # Load pre-trained weights
            in_channels=3,
            classes=num_classes
        )
        print("[*] Initialized U-Net with a pre-trained EfficientNet-B4 encoder.")

    def forward(self, x):
        return self.model(x)