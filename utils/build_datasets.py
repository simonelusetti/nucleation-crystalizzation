#!/usr/bin/env python3
"""
build_datasets.py — Preprocess and encode NER datasets for the nucleation pipeline.

Saves a DatasetDict per source dataset with fields:
  tokens    : List[str]
  binary    : List[int]           1 = entity, 0 = non-entity
  ner_tags  : List[str]           coarse BIO strings  (e.g. "B-per", "I-org", "O")
  encodings : List[List[float]]   BERT hidden states at --layer, one vec per word

Entity type normalisation across datasets
─────────────────────────────────────────
  per          person / PER
  org          organization / ORG / corporation
  loc          location / LOC
  misc         MISC (CoNLL only)
  building     Few-NERD building
  art          Few-NERD art
  product      product (Few-NERD + WNUT)
  event        Few-NERD event
  other        Few-NERD other
  creative-work  WNUT creative-work
  group        WNUT group

Usage
─────
  python build_datasets.py
  python build_datasets.py --model bert-base-cased --layer 8 --batch-size 64
"""

import argparse
from pathlib import Path

import torch
from datasets import Dataset, DatasetDict, load_dataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

OUTPUT_DIR = "data/annotated"


# ── BIO tag coarsening ────────────────────────────────────────────────────────

# Few-NERD coarse type → normalised label
_FN_TYPE = {
    "person":       "per",
    "location":     "loc",
    "organization": "org",
    "building":     "building",
    "art":          "art",
    "product":      "product",
    "event":        "event",
    "other":        "other",
}

def _fn_bio(tag_str: str) -> str:
    """B-person-actor → B-per,  I-location-GPE → I-loc,  O → O."""
    if tag_str == "O":
        return "O"
    prefix = tag_str[:2]                    # "B-" or "I-"
    coarse = tag_str[2:].split("-")[0]      # "person", "location", …
    return prefix + _FN_TYPE.get(coarse, coarse)

_WIKIANN_INT = {0: "O", 1: "B-per", 2: "I-per", 3: "B-org", 4: "I-org", 5: "B-loc", 6: "I-loc"}

# CoNLL and WNUT type normalisation (applied to the type part of existing BIO strings)
_TYPE_NORM = {
    "PER": "per", "ORG": "org", "LOC": "loc", "MISC": "misc",
    "person": "per", "location": "loc", "corporation": "org",
    "product": "product", "creative-work": "creative-work", "group": "group",
}

def _norm_bio(tag_str: str) -> str:
    """Lowercase / normalise the entity-type part of a BIO tag."""
    if tag_str == "O":
        return "O"
    prefix, typ = tag_str[:2], tag_str[2:]
    return prefix + _TYPE_NORM.get(typ, typ.lower())


# ── BERT encoding ─────────────────────────────────────────────────────────────

def _encode(examples: list[dict], tokenizer, model, layer: int, device: str) -> list[list[list[float]]]:
    """
    Run BERT on a batch of pre-tokenised sentences and return first-subword
    hidden states at `layer`.  Returns List[List[List[float]]] — one float
    vector per word per sentence.
    """
    with torch.no_grad():
        enc = tokenizer(
            [ex["tokens"] for ex in examples],
            is_split_into_words=True,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        ).to(device)

        hidden = model(**enc, output_hidden_states=True).hidden_states[layer]
        hidden = hidden.cpu().float()   # (B, seq_len, H)

    H = hidden.shape[-1]
    result = []
    for b, ex in enumerate(examples):
        n_words  = len(ex["tokens"])
        word_ids = enc.word_ids(b)

        first_sub = torch.full((n_words,), -1, dtype=torch.long)
        for tok_i, word_i in enumerate(word_ids):
            if word_i is not None and word_i < n_words and first_sub[word_i] == -1:
                first_sub[word_i] = tok_i

        vecs = torch.zeros(n_words, H)
        valid = first_sub >= 0
        vecs[valid] = hidden[b, first_sub[valid]]

        result.append(vecs.tolist())

    return result


# ── Shared build helper ───────────────────────────────────────────────────────

def _build_splits(
    name: str,
    raw,
    tag_fn,             # int → BIO string
    tokenizer,
    model,
    layer: int,
    device: str,
    batch_size: int,
) -> DatasetDict:
    splits = {}
    for split in ["train", "validation", "test"]:
        raw_split = raw[split]
        n = len(raw_split)

        # Quick stats — read directly from the HF dataset, no Python list built
        n_tok = sum(len(raw_split[i]["tokens"]) for i in range(n))
        n_ent = sum(
            sum(1 for t in raw_split[i]["ner_tags"] if tag_fn(t) != "O")
            for i in range(n)
        )
        print(f"  {name}/{split:<12}  sents={n:>6,}  "
              f"tokens={n_tok:>8,}  entity={n_ent:>7,} ({n_ent/max(n_tok,1):.1%})")

        # Generator — encodes one batch at a time and yields examples one by one.
        # Dataset.from_generator writes directly to Arrow; never holds the full
        # split in RAM simultaneously.
        def _gen(raw_split=raw_split, n=n, split=split):
            for i in tqdm(range(0, n, batch_size), desc=f"  encoding {split}", leave=False):
                batch_slice = [raw_split[j] for j in range(i, min(i + batch_size, n))]
                examples = []
                for row in batch_slice:
                    tags = [tag_fn(t) for t in row["ner_tags"]]
                    examples.append({
                        "tokens":   list(row["tokens"]),
                        "binary":   [0 if t == "O" else 1 for t in tags],
                        "ner_tags": tags,
                    })
                vecs = _encode(examples, tokenizer, model, layer, device)
                for ex, v in zip(examples, vecs):
                    yield {**ex, "encodings": v}

        splits[split] = Dataset.from_generator(_gen)
    return DatasetDict(splits)


# ── Per-dataset builders ──────────────────────────────────────────────────────

def build_fewnerd(tokenizer, model, layer, device, batch_size) -> DatasetDict:
    raw   = load_dataset("DFKI-SLT/few-nerd", "supervised")
    feats = raw["train"].features["ner_tags"]
    if hasattr(feats, "feature"):
        feats = feats.feature
    names  = feats.names if hasattr(feats, "names") else None
    to_str = (lambda t: names[t]) if names else str
    return _build_splits("fewnerd", raw, lambda t: _fn_bio(to_str(t)),
                         tokenizer, model, layer, device, batch_size)


def build_wikiann(tokenizer, model, layer, device, batch_size) -> DatasetDict:
    raw = load_dataset("wikiann", "en")
    return _build_splits("wikiann", raw, lambda t: _WIKIANN_INT[t],
                         tokenizer, model, layer, device, batch_size)


def build_conll2003(tokenizer, model, layer, device, batch_size) -> DatasetDict:
    raw   = load_dataset("conll2003", trust_remote_code=True)
    feats = raw["train"].features["ner_tags"]
    if hasattr(feats, "feature"):
        feats = feats.feature
    names  = feats.names if hasattr(feats, "names") else None
    to_str = (lambda t: names[t]) if names else str
    return _build_splits("conll2003", raw, lambda t: _norm_bio(to_str(t)),
                         tokenizer, model, layer, device, batch_size)


def build_wnut17(tokenizer, model, layer, device, batch_size) -> DatasetDict:
    raw   = load_dataset("wnut_17")
    feats = raw["train"].features["ner_tags"]
    if hasattr(feats, "feature"):
        feats = feats.feature
    names  = feats.names if hasattr(feats, "names") else None
    to_str = (lambda t: names[t]) if names else str
    return _build_splits("wnut17", raw, lambda t: _norm_bio(to_str(t)),
                         tokenizer, model, layer, device, batch_size)


# ── Main ──────────────────────────────────────────────────────────────────────

BUILDERS = {
    "fewnerd":   build_fewnerd,
    "wikiann":   build_wikiann,
    "conll2003": build_conll2003,
    "wnut17":    build_wnut17,
}


def main(args: argparse.Namespace) -> None:
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Model: {args.model}  layer: {args.layer}  "
          f"batch_size: {args.batch_size}  device: {device}\n")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    bert = AutoModel.from_pretrained(args.model).to(device)
    bert.eval()
    for p in bert.parameters():
        p.requires_grad_(False)

    out = Path(args.output_dir)
    for name, builder in BUILDERS.items():
        print(f"\nBuilding {name}…")
        ds   = builder(tokenizer, bert, args.layer, device, args.batch_size)
        path = out / name
        path.mkdir(parents=True, exist_ok=True)
        ds.save_to_disk(str(path))
        print(f"  → saved to {path}/")

    print("\nDone.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model",      default="bert-base-cased")
    p.add_argument("--layer",      type=int, default=8)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--device",     default=None, help="cpu | cuda (default: auto)")
    p.add_argument("--output-dir", default=OUTPUT_DIR)
    main(p.parse_args())
