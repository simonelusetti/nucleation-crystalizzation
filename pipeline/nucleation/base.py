"""Abstract base for nucleation (seed identification) backends."""

from abc import ABC, abstractmethod

import torch
from torch.utils.data import DataLoader


class NucleationBackend(ABC):
    def fit(self, loader: DataLoader) -> "NucleationBackend":
        return self

    @abstractmethod
    def predict(self, loader: DataLoader) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns (preds, lengths):
          preds   : (N, max_len) LongTensor — padded binary predictions
          lengths : (N,) LongTensor — actual sentence lengths
        """
        ...
