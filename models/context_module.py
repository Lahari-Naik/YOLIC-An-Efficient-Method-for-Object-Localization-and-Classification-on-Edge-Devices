"""
Inter-Cell Context Module & Token Projector for YOLIC
Models spatial dependencies across neighboring Cells of Interest (CoIs) via Multi-Head Self-Attention.
"""
import torch
import torch.nn as nn


class TokenProjector(nn.Module):
    """
    Projects the spatial feature map F in R^(B x C_feat x 7 x 7) into N cell tokens:
    H^(0) = ReLU(Linear(Flatten(AdaptiveAvgPool2d_7x7(F)))) in R^(B x N x d_cell)
    where d_cell = 128.
    """
    def __init__(self, in_channels: int, num_cois: int = 104, cell_dim: int = 128):
        super().__init__()
        self.num_cois = num_cois
        self.cell_dim = cell_dim
        # 7x7 spatial pooling over F down to (1, 1) spatial dimension
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten(start_dim=1)
        self.proj = nn.Linear(in_channels, num_cois * cell_dim)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, F: torch.Tensor) -> torch.Tensor:
        # F: (B, C_feat, 7, 7)
        pooled = self.pool(F)           # (B, C_feat, 1, 1)
        flat = self.flatten(pooled)     # (B, C_feat)
        out = self.relu(self.proj(flat)) # (B, N * d_cell)
        B = out.size(0)
        H0 = out.view(B, self.num_cois, self.cell_dim) # (B, N, d_cell)
        return H0


class InterCellContextModule(nn.Module):
    """
    Self-Attention Layer:
    Input: H^(0) in R^(B x N x d_cell)
    Computes MHSA with h = 4 heads, dropout = 0.1.
    Residual connection and LayerNorm:
    H^(1) = LayerNorm(H^(0) + Dropout(MHSA(Q=H^(0), K=H^(0), V=H^(0))))
    """
    def __init__(self, cell_dim: int = 128, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.cell_dim = cell_dim
        self.num_heads = num_heads
        self.mha = nn.MultiheadAttention(
            embed_dim=cell_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(cell_dim)

    def forward(self, H0: torch.Tensor) -> torch.Tensor:
        # H0: (B, N, d_cell)
        attn_out, _ = self.mha(query=H0, key=H0, value=H0)
        H1 = self.layer_norm(H0 + self.dropout(attn_out))
        return H1
