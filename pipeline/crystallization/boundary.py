"""Crystallization via window-based span expansion from seed anchors."""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from .base import CrystallizationBackend


class BoundaryCrystallization(CrystallizationBackend):
    """
    Expand each predicted seed token outward by up to `window` positions.
    Implemented as a single vectorised 1-D max-pool dilation over all sentences.
    """

    def __init__(self, window: int):
        self.window = window

    def expand(
        self,
        loader: DataLoader,
        nuc_preds: torch.Tensor,
        lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        kernel = 2 * self.window + 1
        x   = nuc_preds.float().unsqueeze(1)                                  # (N, 1, max_len)
        out = F.max_pool1d(x, kernel_size=kernel, stride=1, padding=self.window)
        return (out.squeeze(1) >= 0.5).long(), lengths


class SmoothingCrystallization(CrystallizationBackend):
    """
    Fill gaps between seeds separated by at most `smooth_window` tokens.
    Vectorised per-sentence via torch.searchsorted.
    """

    def __init__(self, smooth_window: int):
        self.smooth_window = smooth_window

    def expand(
        self,
        loader: DataLoader,
        nuc_preds: torch.Tensor,
        lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        w      = self.smooth_window
        result = nuc_preds.clone()

        for i in tqdm(range(len(nuc_preds)), desc="smoothing expand", leave=False):
            n        = int(lengths[i])
            x        = nuc_preds[i, :n]
            seed_pos = x.nonzero(as_tuple=True)[0]
            if seed_pos.numel() < 2:
                continue

            pos       = torch.arange(n)
            right_idx = torch.searchsorted(seed_pos, pos)
            left_idx  = right_idx - 1

            INF       = n + w + 1
            right_pos = torch.where(
                right_idx < seed_pos.numel(),
                seed_pos[right_idx.clamp(max=seed_pos.numel() - 1)],
                torch.full_like(pos, INF),
            )
            left_pos  = torch.where(
                left_idx >= 0,
                seed_pos[left_idx.clamp(min=0)],
                torch.full_like(pos, -(w + 1)),
            )

            fill = ((right_pos - left_pos - 1) <= w) & (x == 0)
            result[i, :n] = x.masked_fill(fill, 1)

        return result, lengths
