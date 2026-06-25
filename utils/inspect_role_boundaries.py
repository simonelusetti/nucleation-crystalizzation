"""Inspect words near the seed/connector decision boundary.

Shows which in-vocab words are closest to the score threshold (most likely to
flip category under small λ changes), and which words changed category compared
to the old entity_rate-only rule.

Usage:
    python utils/inspect_role_boundaries.py
    python utils/inspect_role_boundaries.py --lambda-seed 0.4 --lambda-oov 0.6
    python utils/inspect_role_boundaries.py --top 60 --out boundaries.txt
"""

import argparse
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline.data import (
    ALL_DATASETS, EVAL_SPLITS, MIN_COUNT, SEED_THRESHOLD,
    RoleParams, build_brown_freq, compute_global_entity_rates, load_split,
)


def _old_role(entity_rate: float, count: int) -> str:
    """Role under the old entity_rate-only rule (no Brown penalty)."""
    if count < MIN_COUNT:
        return "entity_oov"
    return "seed" if entity_rate >= SEED_THRESHOLD else "connector"


def _new_role_oov(brown_norm: float, lambda_oov: float) -> str:
    return "seed" if brown_norm <= lambda_oov else "connector"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir",    default=str(ROOT / "data/annotated"))
    ap.add_argument("--lambda-seed",    type=float, default=0.3)
    ap.add_argument("--lambda-oov",     type=float, default=0.1)
    ap.add_argument("--threshold",      type=float, default=SEED_THRESHOLD)
    ap.add_argument("--caps-threshold", type=float, default=0.5,
                    help="fraction of uppercase letters needed to treat a sentence as all-caps")
    ap.add_argument("--top",         type=int,   default=50,
                    help="number of most-borderline in-vocab words to show")
    ap.add_argument("--out",         default=None,
                    help="write output to this file instead of stdout")
    args = ap.parse_args()
    threshold = args.threshold

    print("loading Brown corpus...", flush=True)
    brown_freq, max_brown_freq = build_brown_freq()
    rp = RoleParams(
        brown_freq=brown_freq,
        max_brown_freq=max_brown_freq,
        lambda_seed=args.lambda_seed,
        lambda_oov=args.lambda_oov,
        caps_threshold=args.caps_threshold,
    )

    print("computing global entity rates...", flush=True)
    global_rate, global_count = compute_global_entity_rates(args.data_dir)

    lines = []
    def out(s=""):
        lines.append(s)

    # ── In-vocab analysis ────────────────────────────────────────────────────

    invocab = [
        (w, global_rate[w], global_count[w])
        for w, c in global_count.items()
        if c >= MIN_COUNT
    ]

    rows = []
    for word, rate, count in invocab:
        bn    = rp.brown_norm(word)
        score = rp.score(rate, word)
        gap   = score - threshold                # negative = connector side
        old   = _old_role(rate, count)
        new   = "seed" if score >= threshold else "connector"
        rows.append((word, rate, bn, score, gap, count, old, new))

    # Sort by |gap| ascending — smallest gap = most borderline
    rows.sort(key=lambda r: abs(r[4]))

    changed = [(r, "seed→connector") if r[6] == "seed" and r[7] == "connector"
               else (r, "connector→seed") if r[6] == "connector" and r[7] == "seed"
               else None
               for r in rows]
    changed = [x for x in changed if x is not None]

    threshold = args.threshold

    out("=" * 80)
    out(f"ROLE BOUNDARY INSPECTION  (λ_seed={args.lambda_seed}  λ_oov={args.lambda_oov}  threshold={threshold})")
    out("=" * 80)
    out()
    out(f"Formula (in-vocab):  score = entity_rate - λ_seed × brown_norm")
    out(f"  seed      if score ≥ {threshold}")
    out(f"  connector if score <  {threshold}")
    out(f"Formula (OOV):       seed if brown_norm ≤ λ_oov, else connector")
    out()

    # ── Section 1: most borderline in-vocab words ────────────────────────────

    out(f"{'─'*80}")
    out(f"TOP {args.top} MOST BORDERLINE IN-VOCAB WORDS  (closest |score − {SEED_THRESHOLD}|)")
    out(f"{'─'*80}")
    out()

    col_w = (20, 7, 8, 8, 7, 6, 12, 12)
    header = (
        f"{'word':<{col_w[0]}} {'rate':>{col_w[1]}} {'bn':>{col_w[2]}} "
        f"{'score':>{col_w[3]}} {'gap':>{col_w[4]}} {'cnt':>{col_w[5]}} "
        f"{'old':>{col_w[6]}} {'new':>{col_w[7]}}"
    )
    out(header)
    out("-" * sum(col_w) + "-" * (len(col_w) - 1))

    for word, rate, bn, score, gap, count, old, new in rows[:args.top]:
        marker = "  ←" if old != new else ""
        out(
            f"{word:<{col_w[0]}} {rate:>{col_w[1]}.3f} {bn:>{col_w[2]}.3f} "
            f"{score:>{col_w[3]}.3f} {gap:>{col_w[4]:}.3f} {count:>{col_w[5]}} "
            f"{old:>{col_w[6]}} {new:>{col_w[7]}}{marker}"
        )

    out()
    out("  ← marks words that changed category")

    # ── Section 2: category changes ──────────────────────────────────────────

    out()
    out(f"{'─'*80}")
    out(f"CATEGORY CHANGES  (total: {len(changed)})")
    out(f"{'─'*80}")

    for direction in ("seed→connector", "connector→seed"):
        subset = [(r, d) for r, d in changed if d == direction]
        out()
        out(f"  {direction}  ({len(subset)} words):")
        out()
        subset.sort(key=lambda x: abs(x[0][4]))   # closest to boundary first
        for (word, rate, bn, score, gap, count, old, new), _ in subset[:30]:
            out(f"    {word:<22} rate={rate:.3f}  bn={bn:.3f}  score={score:.3f}  cnt={count}")

    # ── Section 3: OOV words near λ_oov boundary ─────────────────────────────

    oov_rows = []
    for word, count in global_count.items():
        if count >= MIN_COUNT:
            continue
        bn     = rp.brown_norm(word)
        gap_oov = bn - args.lambda_oov       # negative = seed side
        entity_c = sum(1 for _ in [])        # we don't have per-word binary here easily
        old = _old_role(0.0, count)          # entity_oov for all OOV entity tokens
        new = _new_role_oov(bn, args.lambda_oov)
        oov_rows.append((word, bn, gap_oov, count, new))

    oov_rows.sort(key=lambda r: abs(r[2]))

    out()
    out(f"{'─'*80}")
    out(f"TOP 40 MOST BORDERLINE OOV WORDS  (closest |brown_norm − λ_oov|={args.lambda_oov})")
    out(f"{'─'*80}")
    out()

    oov_col_w = (25, 8, 8, 6, 12)
    oov_header = (
        f"{'word':<{oov_col_w[0]}} {'bn':>{oov_col_w[1]}} "
        f"{'gap':>{oov_col_w[2]}} {'cnt':>{oov_col_w[3]}} {'new_role':>{oov_col_w[4]}}"
    )
    out(oov_header)
    out("-" * sum(oov_col_w) + "-" * (len(oov_col_w) - 1))

    for word, bn, gap_oov, count, new in oov_rows[:40]:
        out(
            f"{word:<{oov_col_w[0]}} {bn:>{oov_col_w[1]}.3f} "
            f"{gap_oov:>{oov_col_w[2]}.3f} {count:>{oov_col_w[3]}} {new:>{oov_col_w[4]}}"
        )

    # ── Summary ───────────────────────────────────────────────────────────────

    total_invocab   = len(rows)
    n_seed_new      = sum(1 for r in rows if r[7] == "seed")
    n_connector_new = sum(1 for r in rows if r[7] == "connector")
    n_changed       = len(changed)
    n_sc            = sum(1 for _, d in changed if d == "seed→connector")
    n_cs            = sum(1 for _, d in changed if d == "connector→seed")

    out()
    out(f"{'─'*80}")
    out("SUMMARY")
    out(f"{'─'*80}")
    out(f"  In-vocab words: {total_invocab}")
    out(f"    seed (new):      {n_seed_new}  ({100*n_seed_new/total_invocab:.1f}%)")
    out(f"    connector (new): {n_connector_new}  ({100*n_connector_new/total_invocab:.1f}%)")
    out(f"  Category changes: {n_changed}")
    out(f"    seed → connector: {n_sc}")
    out(f"    connector → seed: {n_cs}")
    out()

    text = "\n".join(lines)
    if args.out:
        Path(args.out).write_text(text)
        print(f"written to {args.out}")
    else:
        print(text)


if __name__ == "__main__":
    main()
