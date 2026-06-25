"""Shared pipeline logic — backends, pipeline orchestration, and entry-point helper."""

import logging
from pathlib import Path

import forge
import torch
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from .data import (
    RoleParams, annotate_global_roles, build_brown_freq, build_gazetteer,
    compute_entity_rates, compute_global_entity_rates,
    load_eval_splits, load_split, make_loader, n_eval_splits,
)
from .metrics import role_breakdown

log = logging.getLogger(__name__)


# ── Backend builders ──────────────────────────────────────────────────────────

def build_nucleation(cfg: DictConfig, device: str = "auto"):
    name = cfg.name
    if name == "prototype":
        from .nucleation.prototype import PrototypeNucleation
        return PrototypeNucleation(
            model=cfg.model,
            layer=int(cfg.layer),
            pca_components=int(cfg.pca_components),
            device=device,
        )
    raise ValueError(f"Unknown nucleation backend: {name!r}")


def build_crystallization(cfg: DictConfig, d: int, num_workers: int = 0, batch_size: int = 8):
    name = cfg.name
    if name == "transition":
        from .crystallization.transition import TransitionCrystallization
        return TransitionCrystallization(
            d=d,
            d_k=int(cfg.d_k),
            K=int(cfg.K),
            gamma=float(cfg.gamma),
            lr=float(cfg.lr),
            epochs=int(cfg.epochs),
            train_batch_size=batch_size,
            lambda_mono=float(cfg.lambda_mono),
            lambda_step=float(cfg.lambda_step),
            num_workers=num_workers,
        )
    raise ValueError(f"Unknown crystallization backend: {name!r}")


def build_end2end(cfg: DictConfig):
    raise ValueError(f"Unknown end2end backend: {cfg.name!r}")


# ── Oracle / identity pass-throughs ───────────────────────────────────────────

def _oracle_nuc(ds) -> tuple[torch.Tensor, torch.Tensor]:
    """Return gold binary labels as a padded tensor without deserialising encodings."""
    from torch.nn.utils.rnn import pad_sequence
    light = ds.select_columns(["binary"]) if hasattr(ds, "select_columns") else ds
    seqs  = [torch.as_tensor(ex["binary"], dtype=torch.long)
             for ex in tqdm(light, desc="oracle nuc", leave=False)]
    lengths = torch.tensor([len(s) for s in seqs], dtype=torch.long)
    preds   = pad_sequence(seqs, batch_first=True, padding_value=0)
    return preds, lengths


# ── Pipeline ──────────────────────────────────────────────────────────────────

class Pipeline:
    """Holds fitted backends and drives prediction."""

    def __init__(self, nuc_backend, cry_backend, e2e_backend, label: str, batch_size: int = 32, num_workers: int = 0):
        self._nuc        = nuc_backend
        self._cry        = cry_backend
        self._e2e        = e2e_backend
        self.label       = label
        self._batch_size = batch_size
        self._num_workers = num_workers

    def predict_stages(self, ds) -> dict[str, list[list[int]]]:
        """
        Return predictions keyed by stage name.
          e2e model        → {"e2e": ...}
          nuc only         → {"nuc": ...}
          nuc + cry        → {"nuc": ..., "cry": ...}
        """
        bs = self._batch_size
        nw = self._num_workers
        if self._e2e is not None:
            return {"e2e": self._e2e.predict(make_loader(ds, bs, num_workers=nw))}

        preds, lengths = self._nuc.predict(make_loader(ds, bs, num_workers=nw)) if self._nuc else _oracle_nuc(ds)
        stages = {"nuc": [preds[i, :lengths[i]].tolist() for i in range(len(lengths))]}

        if self._cry:
            preds, lengths = self._cry.expand(make_loader(ds, bs, num_workers=nw), preds, lengths)
            stages["cry"] = [preds[i, :lengths[i]].tolist() for i in range(len(lengths))]

        return stages

    def predict(self, ds) -> list[list[int]]:
        return list(self.predict_stages(ds).values())[-1]


# ── Setup ─────────────────────────────────────────────────────────────────────

def setup(
    cfg: DictConfig,
    source_ds,
    *,
    device: str = "auto",
    num_workers: int = 0,
    batch_size: int = 8,
    on_nuc_ready=None,
) -> Pipeline:
    e2e_name = OmegaConf.select(cfg, "end2end.name")
    nuc_name = OmegaConf.select(cfg, "nucleation.name")
    cry_name = OmegaConf.select(cfg, "crystallization.name")

    if e2e_name:
        log.info("end-to-end: %s", e2e_name)
        e2e = build_end2end(cfg.end2end)
        e2e.fit(make_loader(source_ds, batch_size, num_workers=num_workers))
        return Pipeline(None, None, e2e, label=f"e2e:{e2e_name}", batch_size=batch_size, num_workers=num_workers)

    nuc, cry = None, None
    parts = []

    if nuc_name:
        log.info("nucleation: %s", nuc_name)
        nuc = build_nucleation(cfg.nucleation, device=device)
        nuc.fit(make_loader(source_ds, batch_size, num_workers=num_workers))
        parts.append(nuc_name)
    else:
        log.info("nucleation: oracle (gold labels)")
        parts.append("oracle")

    if cry_name:
        if on_nuc_ready is not None:
            log.info("evaluating nucleation before crystallization training...")
            on_nuc_ready(Pipeline(nuc, None, None, label="+".join(parts),
                                  batch_size=batch_size, num_workers=num_workers))

        log.info("crystallization: %s", cry_name)
        if nuc is not None:
            d = nuc.bert.config.hidden_size
        else:
            from transformers import AutoConfig
            d = AutoConfig.from_pretrained(cfg.nucleation.model).hidden_size
        cry = build_crystallization(cfg.crystallization, d=d, num_workers=num_workers, batch_size=batch_size)
        log.info("generating nucleation predictions for crystallization training...")
        nuc_preds, lengths = nuc.predict(make_loader(source_ds, batch_size, num_workers=num_workers)) if nuc else _oracle_nuc(source_ds)
        cry.fit(make_loader(source_ds, batch_size, num_workers=num_workers), nuc_preds, lengths)
        del nuc_preds, lengths
        parts.append(cry_name)
    else:
        log.info("crystallization: identity (pass-through)")
        parts.append("identity")

    return Pipeline(nuc, cry, None, label="+".join(parts), batch_size=batch_size, num_workers=num_workers)


# ── Evaluation ────────────────────────────────────────────────────────────────

def _metrics_table() -> "PrettyTable":
    from prettytable import PrettyTable
    t = PrettyTable(["dataset/split", "span_f1", "token_f1", "seed_f1", "connector_f1"])
    t.align["dataset/split"] = "l"
    for col in t.field_names[1:]:
        t.align[col] = "r"
    return t


def _fmt(v) -> str:
    return f"{v:.4f}" if v is not None else "—"


def _add_metrics_row(table, key: str, m: dict) -> None:
    table.add_row([key,
        _fmt(m["span_f1"]), _fmt(m["token_f1"]),
        _fmt(m["seed_f1"]), _fmt(m["connector_f1"]),
    ])


def evaluate(
    pipeline: Pipeline,
    cfg: DictConfig,
    entity_rate: dict,
    source_count: dict,
    global_rate: dict,
    global_count: dict,
    rp: RoleParams,
) -> dict[str, float]:
    eval_subset = OmegaConf.select(cfg, "data.eval_subset")
    total    = n_eval_splits(cfg.data.dataset, cfg.data.split)
    eval_gen = load_eval_splits(
        cfg.data.dir, cfg.data.dataset, cfg.data.split,
        entity_rate, source_count,
        global_rate, global_count,
        rp,
        eval_subset=eval_subset,
    )

    stage_names: list[str] = []
    tables: dict[tuple[str, str], "PrettyTable"] = {}
    all_metrics: dict[str, dict] = {}

    for ds_name, split_name, annotated_ds in tqdm(eval_gen, desc="evaluation", unit="split", total=total):
        key = f"{ds_name}/{split_name}"
        log.info("evaluating %s  (%d sentences)", key, len(annotated_ds))

        stage_preds = pipeline.predict_stages(annotated_ds)

        if not stage_names:
            stage_names = list(stage_preds)
            for stage in stage_names:
                tables[(stage, "src")]    = _metrics_table()
                tables[(stage, "global")] = _metrics_table()

        all_metrics[key] = {}
        for stage, preds in stage_preds.items():
            m_src    = role_breakdown(annotated_ds, preds, entity_rate=entity_rate, count=source_count, rp=rp)
            m_global = role_breakdown(annotated_ds, preds, entity_rate=global_rate,  count=global_count, rp=rp)
            all_metrics[key][f"{stage}/src"]    = m_src
            all_metrics[key][f"{stage}/global"] = m_global
            _add_metrics_row(tables[(stage, "src")],    key, m_src)
            _add_metrics_row(tables[(stage, "global")], key, m_global)
        del stage_preds

    label = pipeline.label
    stage_labels = {"nuc": "nucleation", "cry": "crystallization", "e2e": "end-to-end"}
    for stage in stage_names:
        name = stage_labels.get(stage, stage)
        log.info("[%s] %s — source roles\n%s", label, name, tables[(stage, "src")])
        log.info("[%s] %s — global roles\n%s", label, name, tables[(stage, "global")])

    return {
        f"{key}/{stage_view}/{metric}": v
        for key, stage_views in all_metrics.items()
        for stage_view, m in stage_views.items()
        for metric, v in m.items()
    }


# ── Shared entry-point ────────────────────────────────────────────────────────

def run_pipeline(cfg: DictConfig, *, force_nuc=None, force_cry=None) -> None:
    """
    Full pipeline run: load source, fit backends, evaluate, log metrics.

    Parameters
    ----------
    force_nuc : if not None, override nucleation.name (e.g. None → oracle mode)
    force_cry : if not None, override crystallization.name (e.g. None → identity mode)
    """
    OmegaConf.update(cfg, "data.dir", str(Path(cfg.data.dir).resolve()))
    if force_nuc is not None:
        OmegaConf.update(cfg, "nucleation.name", force_nuc, force_add=True)
    if force_cry is not None:
        OmegaConf.update(cfg, "crystallization.name", force_cry, force_add=True)

    device      = str(OmegaConf.select(cfg, "runtime.device", default="auto"))
    num_workers = int(OmegaConf.select(cfg, "runtime.workers", default=0))
    batch_size  = int(OmegaConf.select(cfg, "runtime.batch_size", default=8))
    threads     = OmegaConf.select(cfg, "runtime.threads")
    if threads is not None:
        torch.set_num_threads(int(threads))
    log.info("runtime: device=%s  workers=%d  batch_size=%d  threads=%s",
             device, num_workers, batch_size, threads if threads is not None else "default")

    lambda_seed    = float(OmegaConf.select(cfg, "roles.lambda_seed",    default=0.3))
    lambda_oov     = float(OmegaConf.select(cfg, "roles.lambda_oov",     default=0.1))
    caps_threshold = float(OmegaConf.select(cfg, "roles.caps_threshold", default=0.5))
    log.info("loading Brown corpus for role scoring...")
    brown_freq, max_brown_freq = build_brown_freq()
    log.info("building gazetteer (names + countries)...")
    gazetteer = build_gazetteer()
    rp = RoleParams(brown_freq=brown_freq, max_brown_freq=max_brown_freq,
                    lambda_seed=lambda_seed, lambda_oov=lambda_oov,
                    caps_threshold=caps_threshold, gazetteer=gazetteer)
    log.info("role params: lambda_seed=%.2f  lambda_oov=%.2f  caps_threshold=%.2f  gazetteer=%d entries",
             lambda_seed, lambda_oov, caps_threshold, len(gazetteer))

    run = forge.start_run(cfg)

    train_subset = OmegaConf.select(cfg, "data.train_subset")
    source = load_split(cfg.data.dir, cfg.data.dataset, cfg.data.split, subset=train_subset)
    log.info("source: %s/%s  (%d sentences)", cfg.data.dataset, cfg.data.split, len(source))

    entity_rate, source_count = compute_entity_rates(source)

    log.info("computing global entity rates across all datasets...")
    global_rate, global_count = compute_global_entity_rates(cfg.data.dir)

    def _eval_nuc(nuc_pipeline):
        evaluate(nuc_pipeline, cfg, entity_rate, source_count, global_rate, global_count, rp)

    pipeline = setup(cfg, source, device=device, num_workers=num_workers, batch_size=batch_size,
                     on_nuc_ready=_eval_nuc)
    del source

    metrics = evaluate(pipeline, cfg, entity_rate, source_count, global_rate, global_count, rp)
    run.finish(metrics=metrics)
