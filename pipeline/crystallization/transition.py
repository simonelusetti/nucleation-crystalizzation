"""Crystallization via learned fixed-K probability growth (TransitionGrower)."""

import logging
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from ..data import make_loader

log = logging.getLogger(__name__)


_SEED_ROLES = {"seed"}


def _conditional_y(
    binary: torch.Tensor,
    nuc_preds: torch.Tensor,
    roles: list[list[str]],
    lengths: torch.Tensor,
    oracle: bool = False,
    true_seed_label: int = 1,
    false_seed_label: int = 0,
) -> torch.Tensor:
    """Condition crystallization targets on nucleation output.

    found_seed = seeds predicted by nucleation (or all GT seeds in oracle mode).
    - True found seeds  (binary=1): labeled true_seed_label
    - False found seeds (binary=0): labeled false_seed_label
    - Connectors / missed seeds in spans with ≥1 true found seed: labeled 1
    - Everything else: labeled 0
    """
    y = torch.zeros_like(binary, dtype=torch.float)
    for b in range(binary.shape[0]):
        n = int(lengths[b])
        bin_b  = binary[b, :n]
        nuc_b  = nuc_preds[b, :n]
        role_b = roles[b][:n]
        is_seed = torch.tensor([r in _SEED_ROLES for r in role_b], dtype=torch.bool)

        # found_seed: seeds that nucleation "fired on" (or GT seeds in oracle mode)
        found_seed = is_seed & (bin_b == 1 if oracle else nuc_b == 1)

        # Explicit labels for found seeds
        if oracle:
            y[b, :n][found_seed] = float(true_seed_label)
        else:
            y[b, :n][found_seed & (bin_b == 1)] = float(true_seed_label)
            y[b, :n][found_seed & (bin_b == 0)] = float(false_seed_label)

        # Span activation: spans where a true found seed exists → label remaining tokens 1
        activating = found_seed & (bin_b == 1)
        i = 0
        while i < n:
            if bin_b[i] == 1:
                j = i + 1
                while j < n and bin_b[j] == 1:
                    j += 1
                if activating[i:j].any():
                    for k in range(i, j):
                        if not found_seed[k]:
                            y[b, k] = 1.0
                i = j
            else:
                i += 1
    return y


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

class TransitionCrystallization:
    """Requires examples to carry an 'encodings' field (from build_datasets.py)."""

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
        label_mode: str = "conditional",
        true_seed_label: int = 1,
        false_seed_label: int = 0,
    ):
        # ponytail: label_mode="unconditional" restores old behavior; "conditional_oracle" is an ablation
        if label_mode not in ("conditional", "conditional_oracle", "unconditional"):
            raise ValueError(f"Unknown label_mode: {label_mode!r}")
        self.grower           = CrystallizationGrower(d=d, d_k=d_k, K=K, gamma=gamma)
        self._d               = d
        self._lr              = lr
        self._epochs          = epochs
        self._train_batch     = train_batch_size
        self._lambda_mono     = lambda_mono
        self._lambda_step     = lambda_step
        self._num_workers     = num_workers
        self._label_mode      = label_mode
        self._true_seed_label  = true_seed_label
        self._false_seed_label = false_seed_label

    def fit(
        self,
        loader: DataLoader,
        nuc_preds: torch.Tensor,
        lengths: torch.Tensor,
    ) -> "TransitionCrystallization":
        dev          = next(self.grower.parameters()).device
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
                lengths = batch["lengths"].to(dev)
                if self._label_mode == "unconditional":
                    y = batch["binary"].float().to(dev)
                else:
                    if "source_roles" not in batch:
                        raise RuntimeError(
                            f"label_mode={self._label_mode!r} requires source_roles — "
                            "annotate the source dataset with roles before fitting"
                        )
                    oracle = self._label_mode == "conditional_oracle"
                    y = _conditional_y(
                        batch["binary"], batch["nuc_preds"], batch["source_roles"], batch["lengths"],
                        oracle=oracle,
                        true_seed_label=self._true_seed_label,
                        false_seed_label=self._false_seed_label,
                    ).to(dev)

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

        log.info("crystallization training\n%s", table)
        self.grower.eval()
        return self

    def expand(
        self,
        loader: DataLoader,
        nuc_preds: torch.Tensor,
        lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        dev = next(self.grower.parameters()).device
        ds  = _PairedDataset(loader.dataset, nuc_preds, lengths)
        expand_loader = make_loader(ds, loader.batch_size or 32, num_workers=self._num_workers)

        N, max_len = nuc_preds.shape
        result     = torch.zeros(N, max_len, dtype=torch.long)
        cursor     = 0

        for batch in tqdm(expand_loader, desc="cry eval", leave=False):
            X   = batch["encodings"].to(dev)
            p0  = batch["nuc_preds"].float().to(dev)
            out = self.grower.predict(X, p0)
            B   = out.shape[0]
            result[cursor:cursor + B, :out.shape[1]] = out.cpu()
            cursor += B

        return result, lengths

