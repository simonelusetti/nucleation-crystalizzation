"""Crystallization via learned fixed-K probability growth (TransitionGrower)."""

import logging
import math
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from ..data import make_loader
from .base import CrystallizationBackend

log = logging.getLogger(__name__)


class _PairedDataset(Dataset):
    """Pairs a HF Dataset with a pre-computed nuc_preds tensor (no copy)."""
    def __init__(self, base_ds, nuc_preds: torch.Tensor, lengths: torch.Tensor):
        self._ds       = base_ds
        self._preds    = nuc_preds   # (N, max_len)
        self._lengths  = lengths     # (N,)

    def __len__(self):
        return len(self._ds)

    def __getitem__(self, i):
        row = dict(self._ds[i])
        n   = int(self._lengths[i])
        row["nuc_preds"] = self._preds[i, :n]   # tensor slice — no Python list
        return row


# ── Core module ───────────────────────────────────────────────────────────────

class CrystallizationGrower(nn.Module):
    """
    Fixed-K attention-based probability grower.

    Learnable parameters: W_Q and W_K, both (d+1, d_k).
    At each step the current seed probabilities are concatenated to the token
    embeddings, then a single attention operation spreads probability mass to
    neighbouring tokens.

    Args:
        d     : token embedding dimension (must match encodings field)
        d_k   : key/query projection dimension
        K     : number of growth steps
        gamma : geometric weight for BCE loss (later steps weighted higher)
    """

    def __init__(self, d: int, d_k: int, K: int, gamma: float = 0.9):
        super().__init__()
        self.K     = K
        self.gamma = gamma
        self.d_k   = d_k
        self.W_Q   = nn.Parameter(torch.empty(d + 1, d_k))
        self.W_K   = nn.Parameter(torch.empty(d + 1, d_k))
        nn.init.xavier_uniform_(self.W_Q)
        nn.init.xavier_uniform_(self.W_K)

    def forward(self, X: torch.Tensor, p0: torch.Tensor) -> list[torch.Tensor]:
        """
        Args:
            X  : (B, N, d)  token embeddings — no gradient expected
            p0 : (B, N)     binary seed vector
        Returns:
            list of K tensors [p_1, …, p_K], each (B, N)
        """
        scale = math.sqrt(self.d_k)
        p_t   = p0
        steps = []

        for _ in range(self.K):
            x_cat  = torch.cat([X, p_t.unsqueeze(-1)], dim=-1)   # (B, N, d+1)
            Q      = x_cat @ self.W_Q                             # (B, N, d_k)
            K_mat  = x_cat @ self.W_K                             # (B, N, d_k)
            T      = Q @ K_mat.transpose(-1, -2) / scale          # (B, N, N)
            p_next = torch.sigmoid(T @ p_t.unsqueeze(-1)).squeeze(-1)  # (B, N)
            steps.append(p_next)
            p_t = p_next

        return steps

    def predict(
        self,
        X: torch.Tensor,
        p0: torch.Tensor,
        threshold: float = 0.5,
    ) -> torch.Tensor:
        """Run forward, threshold final p_K. Returns (B, N) long tensor."""
        with torch.no_grad():
            steps = self.forward(X, p0)
        return (steps[-1] >= threshold).long()


# ── Loss function ─────────────────────────────────────────────────────────────

def crystallization_loss(
    p_steps: list[torch.Tensor],
    p0: torch.Tensor,
    y: torch.Tensor,
    lengths: torch.Tensor,
    lambda_mono: float = 1.0,
    lambda_step: float = 0.01,
    gamma: float = 0.9,
) -> tuple[torch.Tensor, dict]:
    """
    Args:
        p_steps    : [p_1, …, p_K] from CrystallizationGrower.forward()
        p0         : (B, N) initial seed vector (step 0)
        y          : (B, N) ground-truth binary labels
        lengths    : (B,) actual sequence lengths (for padding mask)
        lambda_mono: weight for monotonicity penalty
        lambda_step: weight for step-cost penalty
        gamma      : geometric decay for BCE weights (matches grower.gamma)

    Returns:
        (total_loss, {"bce": float, "mono": float, "step": float})
    """
    K   = len(p_steps)
    B   = y.shape[0]
    y_f = y.float()

    mask    = torch.arange(y.shape[1], device=y.device).unsqueeze(0) < lengths.unsqueeze(1)
    mask_f  = mask.float()
    n_valid = mask_f.sum()

    # Term 1 — gamma-weighted masked BCE at every step
    L_bce = p0.new_zeros(())
    for t, p in enumerate(p_steps):
        weight = gamma ** (K - 1 - t)
        bce    = F.binary_cross_entropy(p, y_f, reduction="none")  # (B, N)
        L_bce  = L_bce + weight * (bce * mask_f).sum() / n_valid

    # Term 2 — monotonicity penalty on entity tokens only (padding excluded)
    all_steps = [p0] + p_steps
    L_mono = p0.new_zeros(())
    for t in range(K):
        decrease = torch.clamp(all_steps[t] - all_steps[t + 1], min=0.0)
        L_mono   = L_mono + (y_f * decrease * mask_f).sum()
    L_mono = L_mono / B

    # Term 3 — step cost (L1 change per step, padding excluded)
    L_step = p0.new_zeros(())
    for t in range(K):
        L_step = L_step + ((all_steps[t + 1] - all_steps[t]).abs() * mask_f).sum()
    L_step = L_step / (K * B)

    total = L_bce + lambda_mono * L_mono + lambda_step * L_step
    return total, {"bce": L_bce.item(), "mono": L_mono.item(), "step": L_step.item()}


# ── Pipeline backend wrapper ──────────────────────────────────────────────────

class TransitionCrystallization(CrystallizationBackend):
    """
    Wraps CrystallizationGrower as a CrystallizationBackend.
    Requires examples to carry an 'encodings' field (from build_datasets.py).
    """

    def __init__(
        self,
        d: int,
        d_k: int,
        K: int,
        gamma: float = 0.9,
        lr: float = 1e-3,
        epochs: int = 5,
        train_batch_size: int = 32,
        lambda_mono: float = 1.0,
        lambda_step: float = 0.01,
        num_workers: int = 0,
    ):
        self.grower          = CrystallizationGrower(d=d, d_k=d_k, K=K, gamma=gamma)
        self._d              = d
        self._lr             = lr
        self._epochs         = epochs
        self._train_batch    = train_batch_size
        self._lambda_mono    = lambda_mono
        self._lambda_step    = lambda_step
        self._num_workers    = num_workers

    @property
    def device(self) -> torch.device:
        return next(self.grower.parameters()).device

    def to(self, device) -> "TransitionCrystallization":
        self.grower.to(device)
        return self

    def fit(
        self,
        loader: DataLoader,
        nuc_preds: torch.Tensor,
        lengths: torch.Tensor,
    ) -> "TransitionCrystallization":
        dev          = self.device
        ds           = _PairedDataset(loader.dataset, nuc_preds, lengths)
        train_loader = make_loader(ds, self._train_batch, shuffle=True, num_workers=self._num_workers)

        optimizer = torch.optim.Adam(self.grower.parameters(), lr=self._lr)
        self.grower.train()

        from prettytable import PrettyTable
        cols = ["epoch", "loss", "bce", "mono", "step"]
        table = PrettyTable(cols)
        table.align["epoch"] = "r"
        for c in cols[1:]:
            table.align[c] = "r"

        for epoch in range(self._epochs):
            sums      = {"loss": 0.0, "bce": 0.0, "mono": 0.0, "step": 0.0}
            n_batches = 0

            for batch in tqdm(train_loader, desc=f"cry epoch {epoch + 1}/{self._epochs}", leave=False):
                X       = batch["encodings"].to(dev)
                p0      = batch["nuc_preds"].float().to(dev)
                y       = batch["binary"].float().to(dev)
                lengths = batch["lengths"].to(dev)

                optimizer.zero_grad()
                steps = self.grower(X, p0)
                loss, terms = crystallization_loss(
                    steps, p0, y, lengths,
                    lambda_mono=self._lambda_mono,
                    lambda_step=self._lambda_step,
                    gamma=self.grower.gamma,
                )
                loss.backward()
                optimizer.step()

                sums["loss"] += loss.item()
                for k in terms:
                    sums[k] += terms[k]
                n_batches += 1

            avgs = {k: sums[k] / n_batches for k in sums}
            table.add_row([
                f"{epoch + 1}/{self._epochs}",
                f"{avgs['loss']:.4f}", f"{avgs['bce']:.4f}",
                f"{avgs['mono']:.4f}", f"{avgs['step']:.4f}",
            ])

            models_dir = Path("models")
            models_dir.mkdir(exist_ok=True)
            ckpt = models_dir / f"transition_epoch{epoch + 1}.pt"
            torch.save(self.grower.state_dict(), ckpt)
            log.info("saved checkpoint %s", ckpt)

        log.info("crystallization training\n%s", table)
        self.grower.eval()
        return self

    def expand(
        self,
        loader: DataLoader,
        nuc_preds: torch.Tensor,
        lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        dev   = self.device
        order = torch.argsort(lengths)
        ds    = _PairedDataset(loader.dataset.select(order.tolist()),
                               nuc_preds[order], lengths[order])
        expand_loader = make_loader(ds, loader.batch_size or 32, num_workers=self._num_workers)

        N, max_len = nuc_preds.shape
        result     = torch.zeros(N, max_len, dtype=torch.long)
        cursor     = 0

        for batch in tqdm(expand_loader, desc="cry eval", leave=False):
            X   = batch["encodings"].to(dev)
            p0  = batch["nuc_preds"].float().to(dev)
            out = self.grower.predict(X, p0)   # (B, batch_max_len)
            B   = out.shape[0]
            result[cursor:cursor + B, :out.shape[1]] = out.cpu()
            cursor += B

        inv_order = torch.argsort(order)
        return result[inv_order], lengths


# ── Self-test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    torch.manual_seed(0)

    d, d_k, K = 768, 64, 3
    B, N      = 2, 20

    grower = CrystallizationGrower(d=d, d_k=d_k, K=K)

    X  = torch.randn(B, N, d)
    X.requires_grad_(False)

    p0 = torch.zeros(B, N)
    for b in range(B):
        seed_idx = torch.randperm(N)[:3]
        p0[b, seed_idx] = 1.0

    print("── Forward pass ──")
    steps = grower(X, p0)
    for t, p in enumerate(steps, 1):
        print(f"  p_{t}: {tuple(p.shape)}")

    y     = (torch.rand(B, N) > 0.7).float()
    total, terms = crystallization_loss(steps, p0, y, gamma=grower.gamma)

    print(f"\n── Loss ──")
    print(f"  bce  = {terms['bce']:.4f}")
    print(f"  mono = {terms['mono']:.4f}")
    print(f"  step = {terms['step']:.4f}")
    print(f"  total= {total.item():.4f}")

    print("\n── Backward ──")
    total.backward()
    assert grower.W_Q.grad is not None and grower.W_Q.grad.abs().sum() > 0, "W_Q grad is zero!"
    assert grower.W_K.grad is not None and grower.W_K.grad.abs().sum() > 0, "W_K grad is zero!"
    print(f"  W_Q grad norm: {grower.W_Q.grad.norm():.4f}")
    print(f"  W_K grad norm: {grower.W_K.grad.norm():.4f}")
    print("  OK — all assertions passed")
