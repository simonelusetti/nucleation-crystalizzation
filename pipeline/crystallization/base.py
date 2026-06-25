"""Abstract base for crystallization (span expansion) backends."""

from abc import ABC, abstractmethod

import torch
from torch.utils.data import DataLoader


class CrystallizationBackend(ABC):
    def fit(
        self,
        loader: DataLoader,
        nuc_preds: torch.Tensor,
        lengths: torch.Tensor,
    ) -> "CrystallizationBackend":
        return self

    @abstractmethod
    def expand(
        self,
        loader: DataLoader,
        nuc_preds: torch.Tensor,
        lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns (preds, lengths):
          preds   : (N, max_len) LongTensor — padded binary predictions
          lengths : (N,) LongTensor — same lengths as input (sentences unchanged)
        """
        ...
