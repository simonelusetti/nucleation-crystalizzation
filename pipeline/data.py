"""Data loading and role annotation for the nucleation pipeline."""

import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import torch
from datasets import load_from_disk
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

log = logging.getLogger(__name__)

MIN_COUNT      = 3
SEED_THRESHOLD = 0.5

EVAL_SPLITS = {
    "fewnerd":   ["validation", "test"],
    "wikiann":   ["test"],
    "conll2003": ["test"],
    "wnut17":    ["test"],
}

ALL_DATASETS = list(EVAL_SPLITS)


# ── Role parameters ───────────────────────────────────────────────────────────

@dataclass
class RoleParams:
    """Parameters controlling seed/connector classification via English frequency penalty."""
    brown_freq: dict
    max_brown_freq: float
    lambda_seed: float = 0.3
    lambda_oov: float = 0.1
    caps_threshold: float = 0.5
    gazetteer: frozenset = field(default_factory=frozenset)

    def brown_norm(self, word: str, force_lower: bool = False) -> float:
        if force_lower:
            lookup = word.lower()
        else:
            lookup = word if word.isupper() else word.lower()
        freq = self.brown_freq.get(lookup, 0.0)
        return math.log(freq + 1) / math.log(self.max_brown_freq + 1)

    def score(self, entity_rate: float, word: str, force_lower: bool = False) -> float:
        return entity_rate - self.lambda_seed * self.brown_norm(word, force_lower)


_BROWN_CACHE: tuple | None = None
_GAZETTEER_CACHE: frozenset | None = None


def build_brown_freq() -> tuple[dict, float]:
    """Load NLTK Brown corpus word frequencies (cached after first call)."""
    global _BROWN_CACHE
    if _BROWN_CACHE is None:
        from collections import Counter
        from nltk.corpus import brown
        counts = Counter(w.lower() for w in brown.words())
        total  = sum(counts.values())
        freq   = {w: c / total * 1_000_000 for w, c in counts.items()}
        _BROWN_CACHE = (freq, max(freq.values()))
    return _BROWN_CACHE


def build_gazetteer() -> frozenset:
    """Build a lowercase set of proper names from NLTK names corpus + pycountry."""
    global _GAZETTEER_CACHE
    if _GAZETTEER_CACHE is None:
        words: set[str] = set()

        from nltk.corpus import names as nltk_names
        words.update(w.lower() for w in nltk_names.words())

        import pycountry
        for country in pycountry.countries:
            words.add(country.name.lower())
            if hasattr(country, "common_name"):
                words.add(country.common_name.lower())
            if hasattr(country, "official_name"):
                words.add(country.official_name.lower())
            words.add(country.alpha_2.lower())
            words.add(country.alpha_3.lower())
        for sub in pycountry.subdivisions:
            words.add(sub.name.lower())

        _GAZETTEER_CACHE = frozenset(words)
    return _GAZETTEER_CACHE


# ── Collation ─────────────────────────────────────────────────────────────────

def _collate(batch: list[dict]) -> dict:
    """Pad variable-length sentences into fixed-size tensors for DataLoader."""
    bin_seqs = [torch.tensor(ex["binary"], dtype=torch.long) for ex in batch]
    lengths  = torch.tensor([len(s) for s in bin_seqs], dtype=torch.long)
    binary   = pad_sequence(bin_seqs, batch_first=True, padding_value=0)

    out = {"tokens": [ex["tokens"] for ex in batch], "binary": binary, "lengths": lengths}

    if "encodings" in batch[0]:
        out["encodings"] = pad_sequence(
            [torch.tensor(ex["encodings"], dtype=torch.float32) for ex in batch],
            batch_first=True, padding_value=0.0,
        )

    if "nuc_preds" in batch[0]:
        out["nuc_preds"] = pad_sequence(
            [torch.as_tensor(ex["nuc_preds"], dtype=torch.long) for ex in batch],
            batch_first=True, padding_value=0,
        )

    if "source_roles" in batch[0]:
        out["source_roles"] = [ex["source_roles"] for ex in batch]
    if "global_roles" in batch[0]:
        out["global_roles"] = [ex["global_roles"] for ex in batch]

    return out


def make_loader(ds, batch_size: int, *, shuffle: bool = False, num_workers: int = 0) -> DataLoader:
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, collate_fn=_collate, num_workers=num_workers)


# ── Subset helper ─────────────────────────────────────────────────────────────

def _apply_subset(ds, subset):
    if subset is None:
        return ds
    n = len(ds)
    k = max(1, int(n * subset)) if isinstance(subset, float) else min(int(subset), n)
    return ds.select(range(k))


# ── Loaders ───────────────────────────────────────────────────────────────────

def load_split(data_dir: str, dataset: str, split: str, subset=None):
    """
    Load a split as a memory-mapped HuggingFace Dataset.
    No data is deserialised into Python until rows are actually accessed.
    """
    log.info("loading %s/%s", dataset, split)
    ds = load_from_disk(str(Path(data_dir) / dataset))[split]
    return _apply_subset(ds, subset)


def compute_entity_rates(source) -> tuple[dict, dict]:
    """
    Compute P(entity | word) and total counts from source sentences.
    Uses select_columns to avoid deserialising encodings during iteration.
    """
    count  = defaultdict(int)
    entity = defaultdict(int)
    light  = source.select_columns(["tokens", "binary"]) if hasattr(source, "select_columns") else source
    for ex in light:
        for word, label in zip(ex["tokens"], ex["binary"]):
            count[word]  += 1
            entity[word] += label
    entity_rate = {w: entity[w] / count[w] for w in count}
    return entity_rate, dict(count)


def _is_mostly_caps(tokens: list[str], threshold: float = 0.5) -> bool:
    """True if ≥ threshold of all alphabetical characters in the sentence are uppercase."""
    letters = [c for t in tokens for c in t if c.isalpha()]
    if not letters:
        return False
    return sum(c.isupper() for c in letters) / len(letters) >= threshold


def _force_lower(tokens: list[str], idx: int, mostly_caps: bool) -> bool:
    """True if Brown lookup should use word.lower() for this token.

    Covers two cases beyond the default isupper() guard:
    - the whole sentence is mostly caps
    - the word is title-cased at a grammatical sentence boundary (position 0
      or immediately after a sentence-ending punctuation mark), where the
      capitalisation is structural rather than semantic.
    """
    if mostly_caps:
        return True
    word = tokens[idx]
    if word and word[0].isupper() and not word.isupper():
        if idx == 0:
            return True
        if tokens[idx - 1][-1:] in ".!?":
            return True
    return False


def _role(binary: int, count: int, rate: float, rp: RoleParams, word: str,
          force_lower: bool = False) -> str:
    """
    Classify a token instance.

    In-vocab (count >= MIN_COUNT): seed if score >= SEED_THRESHOLD, else connector.
    OOV (count < MIN_COUNT): seed if rarely English (brown_norm <= lambda_oov), else connector.
    Non-entity tokens (binary == 0) are always non_entity regardless of count.
    force_lower: use word.lower() for the Brown lookup regardless of casing.
    """
    if binary == 0:
        return "non_entity"
    if word.lower() in rp.gazetteer:
        return "seed"
    bn = rp.brown_norm(word, force_lower)
    if count < MIN_COUNT:
        return "seed" if bn <= rp.lambda_oov else "connector"
    score = rate - rp.lambda_seed * bn
    return "seed" if score >= SEED_THRESHOLD else "connector"


def annotate(ds, entity_rate: dict, source_count: dict, rp: RoleParams):
    """
    Add roles, entity_rates, and source_counts columns to a Dataset via map().
    Existing columns (including encodings) are preserved.
    load_from_cache_file=False prevents stale cache when entity_rate changes.
    """
    def _fn(batch):
        all_roles, all_rates, all_counts = [], [], []
        for tokens, binary in zip(batch["tokens"], batch["binary"]):
            mostly_caps = _is_mostly_caps(tokens, rp.caps_threshold)
            roles, rates, counts = [], [], []
            for idx, (word, label) in enumerate(zip(tokens, binary)):
                cnt  = source_count.get(word, 0)
                rate = entity_rate.get(word, -1.0) if cnt >= MIN_COUNT else -1.0
                roles.append(_role(label, cnt, rate, rp, word, _force_lower(tokens, idx, mostly_caps)))
                rates.append(float(rate))
                counts.append(int(cnt))
            all_roles.append(roles)
            all_rates.append(rates)
            all_counts.append(counts)
        return {"source_roles": all_roles, "entity_rates": all_rates, "source_counts": all_counts}

    return ds.map(_fn, batched=True, batch_size=1000, load_from_cache_file=False, desc="annotating source roles")


def compute_global_entity_rates(data_dir: str) -> tuple[dict, dict]:
    """Compute entity rates across every dataset and split combined."""
    count  = defaultdict(int)
    entity = defaultdict(int)
    for ds_name in ALL_DATASETS:
        all_splits = list(dict.fromkeys(["train"] + EVAL_SPLITS.get(ds_name, [])))
        for split in all_splits:
            try:
                ds = load_split(data_dir, ds_name, split)
            except Exception:
                continue
            light = ds.select_columns(["tokens", "binary"]) if hasattr(ds, "select_columns") else ds
            for ex in light:
                for word, label in zip(ex["tokens"], ex["binary"]):
                    count[word]  += 1
                    entity[word] += label
    entity_rate = {w: entity[w] / count[w] for w in count}
    return entity_rate, dict(count)


def annotate_global_roles(ds, global_rate: dict, global_count: dict, rp: RoleParams):
    """Add global_roles column using entity rates from the entire corpus."""
    def _fn(batch):
        all_roles = []
        for tokens, binary in zip(batch["tokens"], batch["binary"]):
            mostly_caps = _is_mostly_caps(tokens, rp.caps_threshold)
            roles = []
            for idx, (word, label) in enumerate(zip(tokens, binary)):
                cnt  = global_count.get(word, 0)
                rate = global_rate.get(word, -1.0) if cnt >= MIN_COUNT else -1.0
                roles.append(_role(label, cnt, rate, rp, word, _force_lower(tokens, idx, mostly_caps)))
            all_roles.append(roles)
        return {"global_roles": all_roles}

    return ds.map(_fn, batched=True, batch_size=1000, load_from_cache_file=False, desc="annotating global roles")


def n_eval_splits(source_dataset: str, source_split: str) -> int:
    return sum(
        1 for ds in ALL_DATASETS for sp in EVAL_SPLITS[ds]
        if not (ds == source_dataset and sp == source_split)
    )


def load_eval_splits(
    data_dir: str,
    source_dataset: str,
    source_split: str,
    entity_rate: dict,
    source_count: dict,
    global_rate: dict,
    global_count: dict,
    rp: RoleParams,
    eval_subset=None,
) -> Iterator:
    """
    Yield (dataset_name, split_name, annotated_Dataset) one at a time.
    Only one split is in RAM at a time; the previous is freed before the next loads.
    """
    for ds_name in ALL_DATASETS:
        for split in EVAL_SPLITS[ds_name]:
            if ds_name == source_dataset and split == source_split:
                continue
            ds = load_split(data_dir, ds_name, split, subset=eval_subset)
            ds = annotate(ds, entity_rate, source_count, rp)
            ds = annotate_global_roles(ds, global_rate, global_count, rp)
            yield ds_name, split, ds
