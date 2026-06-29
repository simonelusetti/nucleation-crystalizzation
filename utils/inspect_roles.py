"""Show seed/connector examples per dataset under presence and variance modes."""

import argparse
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline.data import (
    ALL_DATASETS, build_corpus_stats, build_cross_dataset_entity_rates, load_split,
)
from pipeline.roles import MIN_COUNT, SEED_THRESHOLD, build_brown_freq


def _classify_presence(word: str, dataset_presence: dict) -> str:
    return "seed" if dataset_presence.get(word.lower(), 0) <= 1 else "connector"


def _classify_variance(word: str, cross_dataset_rates: dict, lambda_seed: float) -> str:
    entry = cross_dataset_rates.get(word.lower())
    if entry is None:
        return "seed"  # OOV → seed (rare word)
    mean_r, std_r = entry
    return "seed" if mean_r - lambda_seed * std_r >= SEED_THRESHOLD else "connector"


def _collect_entity_words(data_dir: str, ds_name: str, n: int = 500) -> list[str]:
    """Sample up to n unique entity words from the train split of ds_name."""
    try:
        ds = load_split(data_dir, ds_name, "train")
    except Exception as e:
        print(f"  [skip] {ds_name}: {e}")
        return []
    words: dict[str, int] = defaultdict(int)
    for ex in ds:
        for word, label in zip(ex["tokens"], ex["binary"]):
            if label == 1:
                words[word] += 1
    # return words seen at least MIN_COUNT times, sorted by frequency descending
    frequent = [w for w, c in words.items() if c >= MIN_COUNT]
    frequent.sort(key=lambda w: -words[w])
    return frequent[:n]



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/annotated")
    ap.add_argument("--lambda-seed", type=float, default=0.3)
    ap.add_argument("--n", type=int, default=20, help="examples to show per category")
    args = ap.parse_args()

    data_dir = str(Path(args.data_dir).resolve())

    print("building corpus stats...")
    _, _, dataset_presence = build_corpus_stats(data_dir)
    print("building cross-dataset entity rates...")
    cross_dataset_rates = build_cross_dataset_entity_rates(data_dir)

    for ds_name in ALL_DATASETS:
        print(f"\n{'='*60}")
        print(f"  {ds_name}")
        print(f"{'='*60}")

        entity_words = _collect_entity_words(data_dir, ds_name)
        if not entity_words:
            continue

        for mode in ("presence", "variance"):
            print(f"\n  ── {mode} mode ──")

            seeds, connectors = [], []
            for w in entity_words:
                if mode == "presence":
                    role = _classify_presence(w, dataset_presence)
                    info = f"n_datasets={dataset_presence.get(w.lower(), 0)}"
                else:
                    role = _classify_variance(w, cross_dataset_rates, args.lambda_seed)
                    entry = cross_dataset_rates.get(w.lower())
                    if entry:
                        mean_r, std_r = entry
                        score = mean_r - args.lambda_seed * std_r
                        info = f"mean={mean_r:.2f} std={std_r:.2f} score={score:.2f}"
                    else:
                        info = "OOV"

                if role == "seed":
                    seeds.append((w, info))
                else:
                    connectors.append((w, info))

            def _print_group(title, items):
                sample = items[:args.n]
                print(f"\n  {title} ({len(items)} total, showing {len(sample)}):")
                for w, info in sample:
                    print(f"    {w:<22} {info}")

            _print_group("seeds", seeds)
            _print_group("connectors", connectors)


if __name__ == "__main__":
    main()
