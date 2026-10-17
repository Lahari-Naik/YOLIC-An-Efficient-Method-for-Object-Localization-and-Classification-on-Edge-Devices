"""
Unified YOLIC Architecture
Supports:
1. Base YOLIC (Flat Head, no context module)
2. Grouped Head YOLIC (Grouped classification head, no context module)
3. Extended YOLIC (Inter-cell Context Module MHSA + Grouped Head)
"""
from typing import Dict, List, Optional, Tuple, Union
import torch
import torch.nn as nn

from .backbones import get_backbone
from .context_module import TokenProjector, InterCellContextModule
from .grouped_head import BaselineFlatHead, GroupedClassificationHead


class UnifiedYOLIC(nn.Module):
    """
    Unified YOLIC Model architecture.
    Args:
        num_cois: Number of Cells of Interest (default: 104 for outdoor e-scooter, 256 for cityscapes).
        num_classes: Number of hazard classes (default: 11). Total output classes = num_classes + 1 (background).
        backbone_type: 'shufflenet_v2' or 'mobilenet_v2'.
        cell_dim: Cell token representation dimensionality d_cell (default: 128).
        attention_heads: Number of attention heads h for MHSA (default: 4).
        attention_dropout: Attention dropout rate (default: 0.1).
        use_context_module: Enable Inter-Cell Context Module (MHSA).
        use_grouped_head: Enable parameter-bounded Grouped Classification Head.
        pretrained_backbone: Load ImageNet pretrained backbone weights.
    """
    def __init__(
        self,
        num_cois: int = 104,
        num_classes: int = 11,
        backbone_type: str = "shufflenet_v2",
        cell_dim: int = 128,
        attention_heads: int = 4,
        attention_dropout: float = 0.1,
        use_context_module: bool = True,
        use_grouped_head: bool = True,
        pretrained_backbone: bool = True
    ):
        super().__init__()
        self.num_cois = num_cois
        self.num_classes = num_classes
        self.total_classes = num_classes + 1  # 11 hazards + 1 background
        self.backbone_type = backbone_type.lower()
        self.cell_dim = cell_dim
        self.use_context_module = use_context_module
        self.use_grouped_head = use_grouped_head

        # 1. Backbone Feature Extractor
        self.backbone = get_backbone(self.backbone_type, pretrained=pretrained_backbone)
        in_channels = self.backbone.out_channels

        # 2. Token Projector & Context Module (for Grouped Head setups)
        if self.use_grouped_head:
            self.token_projector = TokenProjector(
                in_channels=in_channels,
                num_cois=num_cois,
                cell_dim=cell_dim
            )
            if self.use_context_module:
                self.context_module = InterCellContextModule(
                    cell_dim=cell_dim,
                    num_heads=attention_heads,
                    dropout=attention_dropout
                )
            else:
                self.context_module = None
            self.head = GroupedClassificationHead(
                cell_dim=cell_dim,
                num_classes=num_classes
            )
        else:
            # Baseline Flat FC Head
            self.token_projector = None
            self.context_module = None
            self.head = BaselineFlatHead(
                in_channels=in_channels,
                num_cois=num_cois,
                num_classes=num_classes,
                dropout=0.2
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Args:
            x: Input image tensor of shape (B, 3, 224, 224)
        Returns:
            Raw logits tensor of shape (B, N, M + 1)
        """
        # Backbone feature extraction: (B, C_feat, 7, 7)
        F = self.backbone(x)

        if self.use_grouped_head:
            # Decompose spatial features into N cell tokens: (B, N, d_cell)
            H0 = self.token_projector(F)
            if self.context_module is not None:
                # Inter-cell self-attention: (B, N, d_cell)
                H = self.context_module(H0)
            else:
                H = H0
            # Shared classification projection: (B, N, M + 1)
            logits = self.head(H)
        else:
            # Baseline Flat Head: (B, N, M + 1)
            logits = self.head(F)

        return logits

    @torch.no_grad()
    def predict_probabilities(self, x: torch.Tensor) -> torch.Tensor:
        """Computes sigmoid probabilities P = sigma(logits) in [0, 1]^(B x N x (M + 1))."""
        logits = self.forward(x)
        return torch.sigmoid(logits)


def build_yolic_from_config(cfg: dict, pretrained: bool = True) -> UnifiedYOLIC:
    """Build UnifiedYOLIC instance from configuration dictionary."""
    model_cfg = cfg.get("model", {})
    return UnifiedYOLIC(
        num_cois=model_cfg.get("num_cois", 104),
        num_classes=model_cfg.get("num_classes", 11),
        backbone_type=model_cfg.get("backbone_type", "shufflenet_v2"),
        cell_dim=model_cfg.get("cell_dim", 128),
        attention_heads=model_cfg.get("attention_heads", 4),
        attention_dropout=model_cfg.get("attention_dropout", 0.1),
        use_context_module=model_cfg.get("use_context_module", True),
        use_grouped_head=model_cfg.get("use_grouped_head", True),
        pretrained_backbone=pretrained
    )
