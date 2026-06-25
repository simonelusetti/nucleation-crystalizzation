"""Evaluation metrics for the nucleation-and-crystallization pipeline."""

import torch

from .data import MIN_COUNT, SEED_THRESHOLD, _force_lower, _is_mostly_caps


def _spans(seq: list[int]) -> set[tuple[int, int]]:
    spans, i, n = set(), 0, len(seq)
    while i < n:
        if seq[i] == 1:
            j = i + 1
            while j < n and seq[j] == 1:
                j += 1
            spans.add((i, j))
            i = j
        else:
            i += 1
    return spans


def span_f1(y_true_seqs: list[list[int]], y_pred_seqs: list[list[int]]) -> float:
    tp = fp = fn = 0
    for yt, yp in zip(y_true_seqs, y_pred_seqs):
        gold = _spans(yt)
        pred = _spans(yp)
        tp += len(gold & pred)
        fp += len(pred - gold)
        fn += len(gold - pred)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0


def _f1(labels: torch.Tensor, preds: torch.Tensor, mask: torch.Tensor) -> float | None:
    yt = labels[mask]
    if not (yt == 1).any():
        return None
    yp = preds[mask]
    tp = ((yt == 1) & (yp == 1)).sum().item()
    fp = ((yt == 0) & (yp == 1)).sum().item()
    fn = ((yt == 1) & (yp == 0)).sum().item()
    pr = tp / (tp + fp) if tp + fp else 0.0
    re = tp / (tp + fn) if tp + fn else 0.0
    return 2 * pr * re / (pr + re) if pr + re else 0.0


def role_breakdown(
    examples,
    predictions: list[list[int]],
    *,
    entity_rate: dict,
    count: dict,
    rp,
) -> dict:
    all_labels, all_preds   = [], []
    is_seed_list, is_conn_list = [], []
    y_true_seqs, y_pred_seqs   = [], []

    for ex, ps in zip(examples, predictions):
        tokens      = ex["tokens"]
        binary      = list(ex["binary"])
        mostly_caps = _is_mostly_caps(tokens, rp.caps_threshold)

        for idx, (word, label) in enumerate(zip(tokens, binary)):
            c  = count.get(word, 0)
            bn = rp.brown_norm(word, _force_lower(tokens, idx, mostly_caps))
            if c < MIN_COUNT:
                role = "seed" if bn <= rp.lambda_oov else "connector"
            else:
                rate  = entity_rate.get(word, 0.0)
                score = rate - rp.lambda_seed * bn
                role  = "seed" if score >= SEED_THRESHOLD else "connector"
            is_seed_list.append(role == "seed")
            is_conn_list.append(role == "connector")
            all_labels.append(label)

        all_preds.extend(ps)
        y_true_seqs.append(binary)
        y_pred_seqs.append(list(ps))

    labels  = torch.tensor(all_labels,   dtype=torch.long)
    preds   = torch.tensor(all_preds,    dtype=torch.long)
    is_seed = torch.tensor(is_seed_list, dtype=torch.bool)
    is_conn = torch.tensor(is_conn_list, dtype=torch.bool)
    all_mask = torch.ones(len(labels),   dtype=torch.bool)

    return {
        "span_f1":      span_f1(y_true_seqs, y_pred_seqs),
        "token_f1":     _f1(labels, preds, all_mask),
        "seed_f1":      _f1(labels, preds, is_seed),
        "connector_f1": _f1(labels, preds, is_conn),
    }
