"""Nucleation via whitened nearest-centroid prototype classifier."""

import logging
import math

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from .base import NucleationBackend

log = logging.getLogger(__name__)


class _Whitener:
    """PCA whitening via torch SVD — equivalent to sklearn PCA(whiten=True)."""

    def fit(self, X: torch.Tensor, n_components: int) -> "_Whitener":
        n = min(n_components, X.shape[0] - 1, X.shape[1])
        self._mu = X.mean(0)
        Xc = X - self._mu
        _, S, Vh = torch.linalg.svd(Xc, full_matrices=False)
        self._components = Vh[:n]
        self._scale = S[:n] / math.sqrt(X.shape[0] - 1)
        return self

    def transform(self, X: torch.Tensor) -> torch.Tensor:
        return (X - self._mu) @ self._components.T / self._scale


class PrototypeNucleation(NucleationBackend):
    """
    BERT (layer N) → PCA whitening → nearest centroid.
    Uses pre-computed encodings from the DataLoader when available; falls
    back to a live BERT forward pass otherwise.
    """

    def __init__(self, model: str, layer: int, pca_components: int, device: str = "auto"):
        self.model_name     = model
        self.layer          = layer
        self.pca_components = pca_components

        device = ("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else device
        self.device    = device
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        bert = AutoModel.from_pretrained(model).to(device)
        bert.eval()
        for p in bert.parameters():
            p.requires_grad_(False)
        self.bert = bert

        self._whitener: _Whitener | None = None
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

    def _extract(self, loader: DataLoader, desc: str = "") -> tuple[torch.Tensor, torch.Tensor]:
        """
        Iterate loader; return (reps, labels) as flat tensors.
        Skips BERT if the batch contains pre-computed 'encodings'.
        """
        all_reps, all_labels = [], []
        use_bert = None

        for batch in tqdm(loader, desc=desc, leave=False):
            lengths = batch["lengths"]
            if use_bert is None:
                use_bert = "encodings" not in batch

            X = self._bert_encode_batch(batch["tokens"], lengths) if use_bert else batch["encodings"]

            mask = torch.arange(X.shape[1]).unsqueeze(0) < lengths.unsqueeze(1)  # (B, max_len)
            all_reps.append(X[mask])
            if "binary" in batch:
                all_labels.append(batch["binary"][mask])

        reps   = torch.cat(all_reps, dim=0)
        labels = torch.cat(all_labels) if all_labels else torch.empty(0, dtype=torch.long)
        return reps, labels

    # ------------------------------------------------------------------

    def fit(self, loader: DataLoader, pca_fit_samples: int = 50_000) -> "PrototypeNucleation":
        log.info("extracting representations for nucleation fit (%d batches)", len(loader))
        reps, labels = self._extract(loader, desc="nucleation fit")

        n = reps.shape[0]
        if n > pca_fit_samples:
            fit_reps = reps[torch.randperm(n)[:pca_fit_samples]]
        else:
            fit_reps = reps

        log.info("fitting PCA whitener on %d / %d tokens", len(fit_reps), n)
        self._whitener = _Whitener().fit(fit_reps, self.pca_components)
        del fit_reps

        reps_w    = self._whitener.transform(reps)
        del reps
        self._pp  = reps_w[labels == 1].mean(0)
        self._np_ = reps_w[labels == 0].mean(0)
        del reps_w, labels
        return self

    # ------------------------------------------------------------------

    def predict(self, loader: DataLoader) -> tuple[torch.Tensor, torch.Tensor]:
        if self._whitener is None:
            raise RuntimeError("Call fit() before predict()")

        sent_preds: list[torch.Tensor] = []
        all_lengths: list[int] = []
        use_bert = None

        for batch in tqdm(loader, desc="nucleation predict", leave=False):
            lengths = batch["lengths"]
            if use_bert is None:
                use_bert = "encodings" not in batch

            X = self._bert_encode_batch(batch["tokens"], lengths) if use_bert else batch["encodings"]

            mask   = torch.arange(X.shape[1]).unsqueeze(0) < lengths.unsqueeze(1)
            flat_X = X[mask]
            flat_w = self._whitener.transform(flat_X)
            d_pos  = ((flat_w - self._pp)  ** 2).sum(1)
            d_neg  = ((flat_w - self._np_) ** 2).sum(1)
            flat_p = (d_pos < d_neg).long()

            sent_preds.extend(torch.split(flat_p, lengths.tolist()))
            all_lengths.extend(lengths.tolist())

        lengths_t = torch.tensor(all_lengths, dtype=torch.long)
        preds_t   = pad_sequence(sent_preds, batch_first=True, padding_value=0)
        return preds_t, lengths_t
