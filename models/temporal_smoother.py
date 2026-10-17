"""
Temporal Consistency Layer for YOLIC
Low-latency Exponential Moving Average (EMA) smoothing across consecutive video frames
to prevent prediction flicker on edge devices.
"""
from typing import Optional, Union
import numpy as np
import torch


class TemporalSmoother:
    """
    Exponential Moving Average (EMA) smoother for frame-wise probabilities:
    S_t = alpha * P_t + (1 - alpha) * S_{t-1}
    where alpha in [0.7, 0.9], default 0.8.
    """
    def __init__(self, alpha: float = 0.8):
        if not (0.0 <= alpha <= 1.0):
            raise ValueError(f"Alpha must be between 0.0 and 1.0, got {alpha}")
        self.alpha = float(alpha)
        self.buffer: Optional[Union[torch.Tensor, np.ndarray]] = None

    def reset(self) -> None:
        """Clears the temporal memory buffer between video streams or test sequences."""
        self.buffer = None

    def __call__(self, probs: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
        """Alias for update."""
        return self.update(probs)

    def update(self, probs: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
        """
        Smooth the incoming probability tensor or array P_t with historical buffer S_{t-1}.
        Args:
            probs: Tensor or array of shape (N, M + 1) or (B, N, M + 1)
        Returns:
            Smoothed probability tensor or array of identical shape.
        """
        if self.buffer is None or (self.buffer.shape != probs.shape):
            # First frame or shape change initializes the buffer
            if isinstance(probs, torch.Tensor):
                self.buffer = probs.detach().clone()
            else:
                self.buffer = np.copy(probs)
            return self.buffer

        if isinstance(probs, torch.Tensor):
            # Ensure buffer is on same device and dtype
            if self.buffer.device != probs.device or self.buffer.dtype != probs.dtype:
                self.buffer = self.buffer.to(device=probs.device, dtype=probs.dtype)
            self.buffer = self.alpha * probs + (1.0 - self.alpha) * self.buffer
            return self.buffer.clone()
        else:
            self.buffer = self.alpha * probs + (1.0 - self.alpha) * self.buffer
            return np.copy(self.buffer)
