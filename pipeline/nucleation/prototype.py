"""Nucleation via whitened nearest-centroid prototype classifier."""

import logging
import math

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

log = logging.getLogger(__name__)


class PrototypeNucleation:
    """
    BERT (layer N) → PCA whitening → nearest centroid.
    Uses pre-computed encodings from the DataLoader when available; falls
    back to a live BERT forward pass otherwise.
    """

    def __init__(self, model: str, layer: int, pca_components: int, device: str = "auto",
                 proto_roles: list[str] | None = None):
        self.model_name     = model
        self.layer          = layer
        self.pca_components = pca_components
        # ponytail: proto_roles=None means "all entity tokens"; upgrade to per-role centroids if needed
        self.proto_roles    = set(proto_roles) if proto_roles else None

        device = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
        self.device    = device
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        bert = AutoModel.from_pretrained(model).to(device)
        bert.eval()
        for p in bert.parameters():
            p.requires_grad_(False)
        self.bert = bert

        self._mu:         torch.Tensor | None = None
        self._components: torch.Tensor | None = None
        self._scale:      torch.Tensor | None = None
        self._pp:  torch.Tensor | None = None
        self._np_: torch.Tensor | None = None

    # ------------------------------------------------------------------

    def _bert_encode_batch(self, tokens: list[list[str]], lengths: torch.Tensor) -> torch.Tensor:
        """Run BERT on one batch; return (B, max_len, H) first-subword reps."""
        enc = self.tokenizer(
            tokens,
            is_split_into_words=True,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        ).to(self.device)
        with torch.no_grad():
            hidden = self.bert(**enc, output_hidden_states=True).hidden_states[self.layer]
            hidden = hidden.cpu().float()   # (B, seq_len, H)

        B, _, H = hidden.shape
        max_len  = int(lengths.max())
        X = torch.zeros(B, max_len, H)
        for b, n in enumerate(lengths.tolist()):
            word_ids  = enc.word_ids(b)
            first_sub = torch.full((n,), -1, dtype=torch.long)
            for tok_i, word_i in enumerate(word_ids):
                if word_i is not None and word_i < n and first_sub[word_i] == -1:
                    first_sub[word_i] = tok_i
            valid = first_sub >= 0
            X[b, :n][valid] = hidden[b, first_sub[valid]]
        return X

    def _get_encodings(self, batch: dict) -> torch.Tensor:
        if "encodings" in batch:
            return batch["encodings"]
        return self._bert_encode_batch(batch["tokens"], batch["lengths"])

    def _extract(self, loader: DataLoader, desc: str = "") -> tuple[torch.Tensor, torch.Tensor, list[str] | None]:
        """Iterate loader; return (reps, labels, flat_roles).

        flat_roles is a list of role strings aligned with reps/labels, or None
        if the batch has no source_roles column.
        """
        all_reps, all_labels, all_roles = [], [], []

        for batch in tqdm(loader, desc=desc, leave=False):
            lengths = batch["lengths"]
            X = self._get_encodings(batch)

            mask = torch.arange(X.shape[1]).unsqueeze(0) < lengths.unsqueeze(1)  # (B, max_len)
            all_reps.append(X[mask])
            if "binary" in batch:
                all_labels.append(batch["binary"][mask])
            if "source_roles" in batch:
                for sent_roles, n in zip(batch["source_roles"], lengths.tolist()):
                    all_roles.extend(sent_roles[:n])

        reps   = torch.cat(all_reps, dim=0)
        labels = torch.cat(all_labels) if all_labels else torch.empty(0, dtype=torch.long)
        return reps, labels, all_roles if all_roles else None

    # ------------------------------------------------------------------

    def _whiten(self, X: torch.Tensor) -> torch.Tensor:
        return (X - self._mu) @ self._components.T / self._scale

    def fit(self, loader: DataLoader, pca_fit_samples: int = 50_000) -> "PrototypeNucleation":
        reps, labels, flat_roles = self._extract(loader, desc="nucleation fit")

        if self.proto_roles is not None:
            if flat_roles is None:
                raise RuntimeError(
                    "nucleation.prototypes is set to a role list but the source dataset has no "
                    "'source_roles' column — annotate the source with roles before fitting"
                )
            pos_mask = torch.tensor([r in self.proto_roles for r in flat_roles], dtype=torch.bool)
        else:
            pos_mask = labels == 1

        n = reps.shape[0]
        fit_reps = reps[torch.randperm(n)[:pca_fit_samples]] if n > pca_fit_samples else reps
        n_comp = min(self.pca_components, fit_reps.shape[0] - 1, fit_reps.shape[1])
        self._mu = fit_reps.mean(0)
        _, S, Vh = torch.linalg.svd(fit_reps - self._mu, full_matrices=False)
        self._components = Vh[:n_comp]
        self._scale = S[:n_comp] / math.sqrt(fit_reps.shape[0] - 1)
        del fit_reps

        reps_w    = self._whiten(reps)
        del reps
        self._pp  = reps_w[pos_mask].mean(0)
        self._np_ = reps_w[labels == 0].mean(0)   # negative always = all non-entity tokens
        del reps_w, labels
        return self

    # ------------------------------------------------------------------

    def predict(self, loader: DataLoader) -> tuple[torch.Tensor, torch.Tensor]:
        if self._mu is None:
            raise RuntimeError("Call fit() before predict()")

        sent_preds: list[torch.Tensor] = []
        all_lengths: list[int] = []

        for batch in tqdm(loader, desc="nucleation predict", leave=False):
            lengths = batch["lengths"]
            X = self._get_encodings(batch)

            mask   = torch.arange(X.shape[1]).unsqueeze(0) < lengths.unsqueeze(1)
            flat_X = X[mask]
            flat_w = self._whiten(flat_X)
            d_pos  = ((flat_w - self._pp)  ** 2).sum(1)
            d_neg  = ((flat_w - self._np_) ** 2).sum(1)
            flat_p = (d_pos < d_neg).long()

            sent_preds.extend(torch.split(flat_p, lengths.tolist()))
            all_lengths.extend(lengths.tolist())

        lengths_t = torch.tensor(all_lengths, dtype=torch.long)
        preds_t   = pad_sequence(sent_preds, batch_first=True, padding_value=0)
        return preds_t, lengths_t
