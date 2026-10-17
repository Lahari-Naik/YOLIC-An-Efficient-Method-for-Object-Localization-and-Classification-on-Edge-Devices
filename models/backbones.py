"""
Backbone architectures for YOLIC Edge Vision:
- ShuffleNetV2 x1.0
- MobileNetV2
Truncated to feature extraction maps before classification heads.
"""
import torch
import torch.nn as nn
import torchvision.models as models
from typing import Tuple


class ShuffleNetV2Backbone(nn.Module):
    """
    ShuffleNetV2 x1.0 feature extractor truncated before the final linear classifier.
    Input: (B, 3, 224, 224)
    Output: F in R^(B x 1024 x 7 x 7)
    """
    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = models.ShuffleNet_V2_X1_0_Weights.DEFAULT if pretrained else None
        try:
            base = models.shufflenet_v2_x1_0(weights=weights)
        except Exception as e:
            print(f"[ShuffleNetV2Backbone] Warning: Could not download pretrained weights ({e}). Initializing randomly.")
            base = models.shufflenet_v2_x1_0(weights=None)
            
        self.conv1 = base.conv1
        self.maxpool = base.maxpool
        self.stage2 = base.stage2
        self.stage3 = base.stage3
        self.stage4 = base.stage4
        self.conv5 = base.conv5
        self.out_channels = 1024
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.conv5(x)
        return x  # (B, 1024, 7, 7)


class MobileNetV2Backbone(nn.Module):
    """
    MobileNetV2 feature extractor truncated to features module.
    Input: (B, 3, 224, 224)
    Output: F in R^(B x 1280 x 7 x 7)
    """
    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
        try:
            base = models.mobilenet_v2(weights=weights)
        except Exception as e:
            print(f"[MobileNetV2Backbone] Warning: Could not download pretrained weights ({e}). Initializing randomly.")
            base = models.mobilenet_v2(weights=None)
            
        self.features = base.features
        self.out_channels = 1280
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)  # (B, 1280, 7, 7)


def get_backbone(backbone_type: str = "shufflenet_v2", pretrained: bool = True) -> nn.Module:
    """Factory function for backbone feature extractors."""
    backbone_type = backbone_type.lower()
    if "shuffle" in backbone_type:
        return ShuffleNetV2Backbone(pretrained=pretrained)
    elif "mobile" in backbone_type:
        return MobileNetV2Backbone(pretrained=pretrained)
    else:
        raise ValueError(f"Unsupported backbone: {backbone_type}. Supported: 'shufflenet_v2', 'mobilenet_v2'")
