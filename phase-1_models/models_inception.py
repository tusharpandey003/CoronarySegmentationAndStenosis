# models_inception.py

import torch.nn as nn
import segmentation_models_pytorch as smp

class InceptionUnetModel(nn.Module):
    """
    A wrapper for a U-Net model with an InceptionResNetV2 backbone.
    This is a powerful, deep CNN architecture known for its excellent performance
    through multi-scale feature processing.
    """
    def __init__(self, num_classes=2):
        super().__init__()
        
        # Instantiate the U-Net model with the specified encoder
        self.model = smp.Unet(
            encoder_name="inceptionresnetv2", # The chosen powerful encoder
            encoder_weights="imagenet",       # Use pre-trained weights
            in_channels=3,
            classes=num_classes
        )
        print("[*] Initialized U-Net with a pre-trained InceptionResNetV2 encoder.")

    def forward(self, x):
        return self.model(x)