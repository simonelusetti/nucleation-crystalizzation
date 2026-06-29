"""Data loading and corpus statistics for the nucleation pipeline."""

import json
import logging
import math
from collections import defaultdict
from pathlib import Path
from typing import Iterator

import torch
from datasets import load_from_disk
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from .roles import (
    MIN_COUNT, RoleParams,
    _force_lower, _is_mostly_caps, _role,
    role_fingerprint,
)

log = logging.getLogger(__name__)

EVAL_SPLITS = {
    "fewnerd":   ["validation", "test"],
    "wikiann":   ["test"],
    "conll2003": ["test"],
    "wnut17":    ["test"],
}

ALL_DATASETS = list(EVAL_SPLITS)


# ── Corpus statistics ─────────────────────────────────────────────────────────

def build_corpus_stats(data_dir: str) -> tuple[dict, dict, dict]:
    """Single pass over all splits of all datasets.

    Returns:
        global_rate      : word -> entity_rate across entire corpus
        global_count     : word -> total token count across entire corpus
        dataset_presence : word.lower() -> n_datasets where it appears as entity
    """
    cache = Path(data_dir).parent / "corpus_stats.json"
    if cache.exists():
        with open(cache) as f:
            d = json.load(f)
        return d["global_rate"], d["global_count"], d["dataset_presence"]

    g_count: dict[str, int] = defaultdict(int)
    g_entity: dict[str, int] = defaultdict(int)
    word_datasets: dict[str, set] = defaultdict(set)
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
                    g_count[word]  += 1
                    g_entity[word] += label
                    if label == 1:
                        word_datasets[word.lower()].add(ds_name)
    global_rate = {w: g_entity[w] / g_count[w] for w in g_count}
    dataset_presence = {w: len(dsets) for w, dsets in word_datasets.items()}
    with open(cache, "w") as f:
        json.dump({"global_rate": global_rate, "global_count": dict(g_count),
                   "dataset_presence": dataset_presence}, f)
    return global_rate, dict(g_count), dataset_presence


def build_cross_dataset_entity_rates(data_dir: str) -> dict:
    """word.lower() -> (mean_entity_rate, std_entity_rate) across train splits.

    Only words meeting MIN_COUNT per dataset are included. Single-dataset words
    get std=0, so variance score reduces to mean_entity_rate >= threshold.
    """
    cache = Path(data_dir).parent / "cross_dataset_rates.json"
    if cache.exists():
        with open(cache) as f:
            return {w: tuple(v) for w, v in json.load(f).items()}

    per_ds: dict[str, dict[str, float]] = {}
    pbar = tqdm(ALL_DATASETS, leave=False)
    for ds_name in pbar:
        pbar.set_description(ds_name + "/train")
        try:
            ds = load_split(data_dir, ds_name, "train")
        except Exception:
            continue
        cnt: dict[str, int] = defaultdict(int)
        ent: dict[str, int] = defaultdict(int)
        light = ds.select_columns(["tokens", "binary"]) if hasattr(ds, "select_columns") else ds
        for ex in light:
            for word, label in zip(ex["tokens"], ex["binary"]):
                w = word.lower()
                cnt[w] += 1
                ent[w] += label
        per_ds[ds_name] = {w: ent[w] / cnt[w] for w in cnt if cnt[w] >= MIN_COUNT}

    all_words: set[str] = set().union(*per_ds.values())
    result: dict[str, tuple[float, float]] = {}
    for word in all_words:
        rates = [per_ds[ds][word] for ds in per_ds if word in per_ds[ds]]
        n = len(rates)
        mean = sum(rates) / n
        std = math.sqrt(sum((r - mean) ** 2 for r in rates) / n) if n > 1 else 0.0
        result[word] = (mean, std)
    with open(cache, "w") as f:
        json.dump(result, f)
    return result


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
    """Load a split as a memory-mapped HuggingFace Dataset."""
    ds = load_from_disk(str(Path(data_dir) / dataset))[split]
    return _apply_subset(ds, subset)


def compute_entity_rates(source) -> tuple[dict, dict]:
    """Compute P(entity | word) and total counts from source sentences."""
    count  = defaultdict(int)
    entity = defaultdict(int)
    light  = source.select_columns(["tokens", "binary"]) if hasattr(source, "select_columns") else source
    for ex in light:
        for word, label in zip(ex["tokens"], ex["binary"]):
            count[word]  += 1
            entity[word] += label
    entity_rate = {w: entity[w] / count[w] for w in count}
    return entity_rate, dict(count)


# ── Role-annotated loaders ────────────────────────────────────────────────────

def load_or_annotate_split(
    data_dir: str,
    eval_dataset: str,
    eval_split: str,
    source_dataset: str,
    entity_rate: dict,
    source_count: dict,
    rp: RoleParams,
    global_rate: dict | None = None,
    global_count: dict | None = None,
    num_proc: int = 1,
):
    """Load a cached annotated split or compute and save it.

    Cache layout (frequency):  roles/{fp}/{eval_dataset}/{eval_split}/{source_dataset}/
    Cache layout (other modes): roles/{fp}/{eval_dataset}/{eval_split}/
    """
    suffix = source_dataset if rp.role_mode == "frequency" else ""
    cache_path = (
        Path(data_dir).parent / "roles"
        / role_fingerprint(rp)
        / eval_dataset / eval_split / suffix
    ).resolve()

    if (cache_path / "dataset_info.json").exists():
        return load_from_disk(str(cache_path))

    log.info("computing roles for %s/%s (src=%s mode=%s) then saving to %s",
             eval_dataset, eval_split, source_dataset, rp.role_mode, cache_path)
    ds = load_split(data_dir, eval_dataset, eval_split)
    use_global = global_rate is not None and global_count is not None

    def _fn(batch, _er=entity_rate, _sc=source_count,
            _gr=global_rate, _gc=global_count, _rp=rp):
        src_roles = []
        glb_roles = [] if use_global else None
        for tokens, binary in zip(batch["tokens"], batch["binary"]):
            mostly_caps = _is_mostly_caps(tokens, _rp.caps_threshold)
            s_r = []
            g_r = [] if use_global else None
            for idx, (word, label) in enumerate(zip(tokens, binary)):
                fl = _force_lower(tokens, idx, mostly_caps)
                sc_ = _sc.get(word, 0)
                sr_ = _er.get(word, -1.0) if sc_ >= MIN_COUNT else -1.0
                s_r.append(_role(label, sc_, sr_, _rp, word, fl))
                if use_global:
                    gc_ = _gc.get(word, 0)
                    gr_ = _gr.get(word, -1.0) if gc_ >= MIN_COUNT else -1.0
                    g_r.append(_role(label, gc_, gr_, _rp, word, fl))
            src_roles.append(s_r)
            if use_global:
                glb_roles.append(g_r)
        out = {"source_roles": src_roles}
        if use_global:
            out["global_roles"] = glb_roles
        return out

    ds = ds.map(_fn, batched=True, batch_size=1000, load_from_cache_file=False,
                num_proc=num_proc, desc=f"annotating {eval_dataset}/{eval_split}")
    ds.save_to_disk(str(cache_path))
    return ds


def role_caches_complete(
    data_dir: str,
    rp: "RoleParams",
    source_dataset: str,
    source_split: str,
    include_source: bool = False,
) -> bool:
    """True when every annotation cache that would be written is already on disk."""
    suffix = source_dataset if rp.role_mode == "frequency" else ""
    roles_dir = Path(data_dir).parent / "roles" / role_fingerprint(rp)

    def _ok(ds, split):
        return (roles_dir / ds / split / suffix / "dataset_info.json").exists()

    if include_source and not _ok(source_dataset, source_split):
        return False
    return all(
        _ok(ds, split)
        for ds in ALL_DATASETS
        for split in EVAL_SPLITS.get(ds, [])
        if not (ds == source_dataset and split == source_split)
    )


def load_eval_splits(
    data_dir: str,
    source_dataset: str,
    source_split: str,
    entity_rate: dict,
    source_count: dict,
    rp: RoleParams,
    global_rate: dict | None = None,
    global_count: dict | None = None,
    eval_subset=None,
    num_proc: int = 1,
) -> Iterator:
    """Yield (dataset_name, split_name, annotated_Dataset) one at a time.

    Annotated datasets are cached to disk and reloaded on subsequent runs.
    eval_subset is applied after loading so the cache stores the full split.
    """
    pairs = [
        (ds_name, split)
        for ds_name in ALL_DATASETS
        for split in EVAL_SPLITS[ds_name]
        if not (ds_name == source_dataset and split == source_split)
    ]
    pbar = tqdm(pairs, unit="split", leave=False)
    for ds_name, split in pbar:
        pbar.set_description(f"{ds_name}/{split}")
        ds = load_or_annotate_split(
            data_dir, ds_name, split, source_dataset,
            entity_rate, source_count, rp,
            global_rate=global_rate, global_count=global_count,
            num_proc=num_proc,
        )
        if eval_subset is not None:
            ds = _apply_subset(ds, eval_subset)
        yield ds_name, split, ds
