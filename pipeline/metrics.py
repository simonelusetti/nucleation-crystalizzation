"""Evaluation metrics for the nucleation-and-crystallization pipeline."""

import torch


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


def _f1(labels: torch.Tensor, preds: torch.Tensor, mask: torch.Tensor | None = None) -> float | None:
    yt = labels if mask is None else labels[mask]
    if not (yt == 1).any():
        return None
    yp = preds if mask is None else preds[mask]
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
    role_col: str,
) -> dict:
    """Compute F1 scores broken down by token role.

    Reads pre-computed roles from examples[role_col] (a column written by
    load_or_annotate_split). Roles are not recomputed here.
    """
    all_labels, all_preds = [], []
    is_seed_list, is_conn_list = [], []
    y_true_seqs, y_pred_seqs = [], []

    all_binary = examples["binary"]
    all_roles  = examples[role_col]

    for binary, roles, ps in zip(all_binary, all_roles, predictions):
        binary = list(binary)
        for label, role in zip(binary, roles):
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

    n_total_entities = (labels == 1).sum().item()
    n_seed_entities  = (labels[is_seed] == 1).sum().item() if is_seed.any() else 0
    seed_coverage = n_seed_entities / n_total_entities if n_total_entities > 0 else None

    return {
        "span_f1":       span_f1(y_true_seqs, y_pred_seqs),
        "token_f1":      _f1(labels, preds),
        "seed_f1":       _f1(labels, preds, is_seed),
        "seed_coverage": seed_coverage,
        "connector_f1":  _f1(labels, preds, is_conn),
    }
