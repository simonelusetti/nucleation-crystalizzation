"""Check structural properties of entity spans w.r.t. seed/connector roles.

Two checks:
  1. Connector-only spans  — no seed token anywhere in the span
  2. Multi-seed spans      — seed tokens appear in 2+ non-contiguous regions

Usage:
    python utils/inspect_span_roles.py
    python utils/inspect_span_roles.py --examples 5 --out span_roles.txt
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline.data import (
    ALL_DATASETS, EVAL_SPLITS, MIN_COUNT, RoleParams,
    _force_lower, _is_mostly_caps, _role,
    build_brown_freq, build_gazetteer, compute_global_entity_rates, load_split,
)


def find_spans(binary: list[int]) -> list[tuple[int, int]]:
    spans, i = [], 0
    while i < len(binary):
        if binary[i] == 1:
            j = i + 1
            while j < len(binary) and binary[j] == 1:
                j += 1
            spans.append((i, j))
            i = j
        else:
            i += 1
    return spans


def count_seed_regions(roles: list[str]) -> int:
    n, in_seed = 0, False
    for r in roles:
        if r == "seed":
            if not in_seed:
                n += 1
                in_seed = True
        else:
            in_seed = False
    return n


def classify_span(roles: list[str]) -> str:
    n_seed = count_seed_regions(roles)
    if n_seed == 0:
        return "connector_only"
    if n_seed > 1:
        return "multi_seed"
    return "ok"


def span_roles(tokens, binary, global_rate, global_count, rp):
    mostly_caps = _is_mostly_caps(tokens, rp.caps_threshold)
    roles = []
    for idx, (word, label) in enumerate(zip(tokens, binary)):
        cnt  = global_count.get(word, 0)
        rate = global_rate.get(word, -1.0) if cnt >= MIN_COUNT else -1.0
        roles.append(_role(label, cnt, rate, rp, word, _force_lower(tokens, idx, mostly_caps)))
    return roles


def fmt_span(tokens, binary, roles, start, end) -> str:
    parts = []
    for i, (tok, lab, rol) in enumerate(zip(tokens, binary, roles)):
        if start <= i < end:
            tag = "S" if rol == "seed" else "C"
            parts.append(f"[{tok}:{tag}]")
        else:
            parts.append(tok)
    return " ".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir",    default=str(ROOT / "data/annotated"))
    ap.add_argument("--lambda-seed", type=float, default=0.3)
    ap.add_argument("--lambda-oov",  type=float, default=0.1)
    ap.add_argument("--caps-threshold", type=float, default=0.5)
    ap.add_argument("--no-gazetteer", action="store_true",
                    help="disable gazetteer hard-override (for comparison)")
    ap.add_argument("--examples",    type=int,   default=8,
                    help="number of example sentences to show per issue type")
    ap.add_argument("--out",         default=None)
    args = ap.parse_args()

    print("loading Brown corpus...", flush=True)
    brown_freq, max_brown_freq = build_brown_freq()
    gazetteer = frozenset() if args.no_gazetteer else build_gazetteer()
    if not args.no_gazetteer:
        print(f"gazetteer: {len(gazetteer)} entries", flush=True)
    rp = RoleParams(brown_freq=brown_freq, max_brown_freq=max_brown_freq,
                    lambda_seed=args.lambda_seed, lambda_oov=args.lambda_oov,
                    caps_threshold=args.caps_threshold, gazetteer=gazetteer)

    print("computing global entity rates...", flush=True)
    global_rate, global_count = compute_global_entity_rates(args.data_dir)

    # counters[ds/split][kind] = count
    # examples[ds/split][kind] = list of formatted strings
    counters = defaultdict(lambda: defaultdict(int))
    examples = defaultdict(lambda: defaultdict(list))

    for ds_name in ALL_DATASETS:
        all_splits = list(dict.fromkeys(["train"] + EVAL_SPLITS.get(ds_name, [])))
        for split in all_splits:
            try:
                ds = load_split(args.data_dir, ds_name, split)
            except Exception:
                continue
            key = f"{ds_name}/{split}"
            print(f"  scanning {key} ({len(ds)} sentences)...", flush=True)

            for ex in ds:
                tokens = ex["tokens"]
                binary = list(ex["binary"])
                roles  = span_roles(tokens, binary, global_rate, global_count, rp)

                for idx, (label, role) in enumerate(zip(binary, roles)):
                    if label == 1:
                        counters[key]["entity_tokens"] += 1
                        if role == "seed":
                            counters[key]["entity_seeds"] += 1
                        else:
                            counters[key]["entity_connectors"] += 1

                for start, end in find_spans(binary):
                    span_r = roles[start:end]
                    kind   = classify_span(span_r)
                    counters[key]["total"] += 1
                    if kind != "ok":
                        counters[key][kind] += 1
                        if len(examples[key][kind]) < args.examples:
                            examples[key][kind].append(
                                fmt_span(tokens, binary, roles, start, end)
                            )

    lines = []
    def out(s=""):
        lines.append(s)

    out("=" * 80)
    out("SPAN ROLE STRUCTURE CHECK")
    out("=" * 80)
    out()

    total_spans   = sum(v["total"]          for v in counters.values())
    total_co      = sum(v["connector_only"] for v in counters.values())
    total_ms      = sum(v["multi_seed"]     for v in counters.values())
    total_etok    = sum(v["entity_tokens"]  for v in counters.values())
    total_eseed   = sum(v["entity_seeds"]   for v in counters.values())
    total_econn   = sum(v["entity_connectors"] for v in counters.values())

    out(f"Total spans:          {total_spans}")
    out(f"Connector-only:       {total_co}  ({100*total_co/total_spans:.2f}%)" if total_spans else "")
    out(f"Multi-seed regions:   {total_ms}  ({100*total_ms/total_spans:.2f}%)" if total_spans else "")
    out()
    out(f"Entity tokens:        {total_etok}")
    out(f"  seeds:              {total_eseed}  ({100*total_eseed/total_etok:.1f}%)" if total_etok else "")
    out(f"  connectors:         {total_econn}  ({100*total_econn/total_etok:.1f}%)" if total_etok else "")
    out()

    col = 30
    out("─" * 80)
    out(f"  {'dataset/split':<{col}} {'spans':>7} {'conn-only':>10} {'multi-seed':>11} {'etok seeds':>11} {'etok conn':>10}")
    out(f"  {'-'*col} {'-------':>7} {'----------':>10} {'-----------':>11} {'-----------':>11} {'----------':>10}")
    for key in sorted(counters):
        v  = counters[key]
        t  = v["total"]
        co = v["connector_only"]
        ms = v["multi_seed"]
        et = v["entity_tokens"]
        es = v["entity_seeds"]
        ec = v["entity_connectors"]
        seed_pct = f"{100*es/et:.1f}%" if et else " —"
        conn_pct = f"{100*ec/et:.1f}%" if et else " —"
        out(f"  {key:<{col}} {t:>7} {co:>8} ({100*co/t:4.1f}%) {ms:>8} ({100*ms/t:4.1f}%)  {seed_pct:>10}  {conn_pct:>9}")

    out()
    out("S = seed token,  C = connector token")

    for key in sorted(counters):
        ds_ex = examples[key]
        if not ds_ex:
            continue
        out()
        out("=" * 80)
        out(f"  {key}")
        out("=" * 80)
        for kind, label in [("connector_only", "connector-only"),
                             ("multi_seed",     "multi-seed")]:
            if not ds_ex[kind]:
                continue
            out()
            out(f"  -- {label} --")
            for line in ds_ex[kind]:
                out(f"    {line}")

    text = "\n".join(lines)
    if args.out:
        Path(args.out).write_text(text)
        print(f"written to {args.out}")
    else:
        print(text)


if __name__ == "__main__":
    main()
