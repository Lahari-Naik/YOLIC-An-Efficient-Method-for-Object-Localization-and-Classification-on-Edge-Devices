"""
Classification Heads for YOLIC:
1. Baseline Flat Head: Scales O(C_feat * N * (M + 1))
2. Proposed Grouped / Parameter-Bounded Head: Bounded at d_cell * (M + 1) + (M + 1), invariant to N.
"""
import torch
import torch.nn as nn


class BaselineFlatHead(nn.Module):
    """
    Baseline Flat FC Head from standard YOLIC:
    F -> GAP -> Dropout(0.2) -> Linear(C_feat, N * (M + 1))
    Parameter count: C_feat * N * (M + 1) + N * (M + 1)
    """
    def __init__(self, in_channels: int, num_cois: int = 104, num_classes: int = 11, dropout: float = 0.2):
        super().__init__()
        self.num_cois = num_cois
        self.num_classes = num_classes
        self.total_classes = num_classes + 1  # Hazards + Background
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(in_channels, num_cois * self.total_classes)

    def forward(self, F: torch.Tensor) -> torch.Tensor:
        # F: (B, C_feat, 7, 7)
        pooled = self.gap(F)           # (B, C_feat, 1, 1)
        flat = torch.flatten(pooled, 1) # (B, C_feat)
        dropped = self.dropout(flat)
        out = self.fc(dropped)          # (B, N * (M + 1))
        B = out.size(0)
        logits = out.view(B, self.num_cois, self.total_classes) # (B, N, M + 1)
        return logits


class GroupedClassificationHead(nn.Module):
    """
    Proposed Grouped / Parameter-Bounded Classification Head:
    Applies a single shared projection W_cls in R^(d_cell x (M + 1)) across each cell token H in R^(B x N x d_cell):
    Logits = H W_cls + b in R^(B x N x (M + 1))
    Head parameter count is strictly d_cell * (M + 1) + (M + 1), completely invariant to N!
    """
    def __init__(self, cell_dim: int = 128, num_classes: int = 11):
        super().__init__()
        self.cell_dim = cell_dim
        self.num_classes = num_classes
        self.total_classes = num_classes + 1  # 11 hazard classes + 1 background class
        # Shared linear projection applied to every cell token
        self.shared_linear = nn.Linear(cell_dim, self.total_classes)

    def forward(self, H: torch.Tensor) -> torch.Tensor:
        # H: (B, N, d_cell)
        # Shared linear layer processes the last dimension across all batches and cells
        logits = self.shared_linear(H)  # (B, N, total_classes)
        return logits
