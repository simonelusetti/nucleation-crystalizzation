"""Shared pipeline logic — backends, pipeline orchestration, and entry-point helper."""

import logging
from pathlib import Path

import forge
import torch
from datasets import disable_caching, disable_progress_bars
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

disable_caching()
disable_progress_bars()

from .data import (
    build_corpus_stats, build_cross_dataset_entity_rates,
    compute_entity_rates,
    load_eval_splits, load_or_annotate_split, load_split, make_loader,
    role_caches_complete,
)
from .roles import RoleParams, build_brown_freq
from .metrics import role_breakdown

log = logging.getLogger(__name__)


# ── Backend builders ──────────────────────────────────────────────────────────

def build_nucleation(cfg: DictConfig, device: str = "auto", proto_roles: list[str] | None = None):
    name = cfg.name
    if name == "prototype":
        from .nucleation.prototype import PrototypeNucleation
        return PrototypeNucleation(
            model=cfg.model,
            layer=int(cfg.layer),
            pca_components=int(cfg.pca_components),
            device=device,
            proto_roles=proto_roles,
        )
    raise ValueError(f"Unknown nucleation backend: {name!r}")


def build_crystallization(cfg: DictConfig, d: int, num_workers: int = 0, batch_size: int = 8):
    name = cfg.name
    if name == "transition":
        from .crystallization.transition import TransitionCrystallization
        lm = OmegaConf.select(cfg, "label_mode")
        if lm is None or isinstance(lm, str):
            lm_name, true_seed_label, false_seed_label = str(lm or "conditional"), 1, 0
        else:
            lm_name          = str(OmegaConf.select(lm, "name", default="conditional"))
            true_seed_label  = 0 if str(OmegaConf.select(lm, "true_seeds",  default="positive")) == "negative" else 1
            false_seed_label = 1 if str(OmegaConf.select(lm, "false_seeds", default="negative")) == "positive" else 0
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
            label_mode=lm_name,
            true_seed_label=true_seed_label,
            false_seed_label=false_seed_label,
        )
    raise ValueError(f"Unknown crystallization backend: {name!r}")


# ── Oracle / identity pass-throughs ───────────────────────────────────────────

def _oracle_nuc(ds) -> tuple[torch.Tensor, torch.Tensor]:
    """Return seed-only predictions when roles are available, else gold binary labels."""
    from torch.nn.utils.rnn import pad_sequence
    has_roles = hasattr(ds, "column_names") and "source_roles" in ds.column_names
    cols  = ["binary", "source_roles"] if has_roles else ["binary"]
    light = ds.select_columns(cols) if hasattr(ds, "select_columns") else ds
    seqs  = []
    for ex in tqdm(light, desc="oracle nuc", leave=False):
        if has_roles:
            pred = [1 if r == "seed" else 0 for r in ex["source_roles"]]
        else:
            pred = ex["binary"]
        seqs.append(torch.as_tensor(pred, dtype=torch.long))
    lengths = torch.tensor([len(s) for s in seqs], dtype=torch.long)
    preds   = pad_sequence(seqs, batch_first=True, padding_value=0)
    return preds, lengths


# ── Pipeline ──────────────────────────────────────────────────────────────────

class Pipeline:
    """Holds fitted backends and drives prediction."""

    def __init__(self, nuc_backend, cry_backend, label: str, batch_size: int = 32, num_workers: int = 0):
        self._nuc        = nuc_backend
        self._cry        = cry_backend
        self.label       = label
        self._batch_size = batch_size
        self._num_workers = num_workers

    def predict_stages(self, ds) -> dict[str, list[list[int]]]:
        bs = self._batch_size
        nw = self._num_workers
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
    proto_roles: list[str] | None = None,
) -> Pipeline:
    nuc_name = OmegaConf.select(cfg, "nucleation.name")
    cry_name = OmegaConf.select(cfg, "crystallization.name")
    nuc, cry = None, None
    parts = []

    if nuc_name:
        log.info("nucleation: %s", nuc_name)
        nuc = build_nucleation(cfg.nucleation, device=device, proto_roles=proto_roles)
        nuc.fit(make_loader(source_ds, batch_size, num_workers=num_workers))
        parts.append(nuc_name)
    else:
        log.info("nucleation: oracle")
        parts.append("oracle")

    if cry_name:
        if on_nuc_ready is not None:
            log.info("evaluating nucleation")
            on_nuc_ready(Pipeline(nuc, None, label="+".join(parts),
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
        log.info("crystallization: none")
        parts.append("identity")

    return Pipeline(nuc, cry, label="+".join(parts), batch_size=batch_size, num_workers=num_workers)


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(
    pipeline: Pipeline,
    cfg: DictConfig,
    data_dir: str,
    entity_rate: dict,
    source_count: dict,
    rp: RoleParams,
    global_rate: dict | None = None,
    global_count: dict | None = None,
) -> dict[str, float]:
    from prettytable import PrettyTable

    use_global = global_rate is not None and global_count is not None

    def _new_table():
        t = PrettyTable(["dataset/split", "span_f1", "token_f1", "seed_f1", "seed_cov", "connector_f1"])
        t.align["dataset/split"] = "l"
        for col in t.field_names[1:]:
            t.align[col] = "r"
        return t

    def _fmt(v):
        return f"{v:.4f}" if v is not None else "—"

    def _add_row(table, key, m):
        table.add_row([key, _fmt(m["span_f1"]), _fmt(m["token_f1"]),
                       _fmt(m["seed_f1"]), _fmt(m["seed_coverage"]),
                       _fmt(m["connector_f1"])])

    eval_subset = OmegaConf.select(cfg, "data.eval_subset")
    num_proc    = max(1, int(OmegaConf.select(cfg, "runtime.workers", default=1)))
    eval_gen = load_eval_splits(
        data_dir, cfg.data.dataset, cfg.data.split,
        entity_rate, source_count, rp,
        global_rate=global_rate, global_count=global_count,
        eval_subset=eval_subset,
        num_proc=num_proc,
    )

    stage_names: list[str] = []
    tables: dict[tuple[str, str], PrettyTable] = {}
    all_metrics: dict[str, dict] = {}

    pbar = tqdm(eval_gen, desc="evaluation", unit="split")
    for ds_name, split_name, annotated_ds in pbar:
        key = f"{ds_name}/{split_name}"
        pbar.set_description(key)
        stage_preds = pipeline.predict_stages(annotated_ds)

        if not stage_names:
            stage_names = list(stage_preds)
            for stage in stage_names:
                tables[(stage, "src")] = _new_table()
                if use_global:
                    tables[(stage, "global")] = _new_table()

        all_metrics[key] = {}
        for stage, preds in stage_preds.items():
            m_src = role_breakdown(annotated_ds, preds, role_col="source_roles")
            all_metrics[key][f"{stage}/src"] = m_src
            _add_row(tables[(stage, "src")], key, m_src)
            if use_global:
                m_global = role_breakdown(annotated_ds, preds, role_col="global_roles")
                all_metrics[key][f"{stage}/global"] = m_global
                _add_row(tables[(stage, "global")], key, m_global)
        del stage_preds

    label = pipeline.label
    stage_labels = {"nuc": "nucleation", "cry": "crystallization", "e2e": "end-to-end"}
    for stage in stage_names:
        name = stage_labels.get(stage, stage)
        log.info("[%s] %s — source roles\n%s", label, name, tables[(stage, "src")])
        if use_global:
            log.info("[%s] %s — global roles\n%s", label, name, tables[(stage, "global")])

    return {
        f"{key}/{stage_view}/{metric}": v
        for key, stage_views in all_metrics.items()
        for stage_view, m in stage_views.items()
        for metric, v in m.items()
    }


# ── Shared entry-point ────────────────────────────────────────────────────────

def run_pipeline(cfg: DictConfig) -> None:
    """Full pipeline run: load source, fit backends, evaluate, log metrics."""
    project_root = Path.cwd()
    run = forge.start_run(cfg)

    data_dir    = str((project_root / cfg.data.dir).resolve())
    device      = str(OmegaConf.select(cfg, "runtime.device", default="auto"))
    num_workers = int(OmegaConf.select(cfg, "runtime.workers", default=0))
    batch_size  = int(OmegaConf.select(cfg, "runtime.batch_size", default=8))
    threads     = OmegaConf.select(cfg, "runtime.threads")
    if threads is not None:
        torch.set_num_threads(int(threads))
    lambda_seed    = float(OmegaConf.select(cfg, "roles.lambda_seed",    default=0.3))
    lambda_oov     = float(OmegaConf.select(cfg, "roles.lambda_oov",     default=0.1))
    caps_threshold = float(OmegaConf.select(cfg, "roles.caps_threshold", default=0.5))
    role_mode      = str(OmegaConf.select(cfg, "roles.mode",             default="frequency"))

    proto_cfg = OmegaConf.select(cfg, "nucleation.prototypes", default="entity")
    proto_roles = None if proto_cfg == "entity" else list(OmegaConf.to_container(proto_cfg))

    # Build a thin RoleParams (no corpus data) sufficient for cache-path fingerprinting.
    rp = RoleParams(brown_freq={}, max_brown_freq=1.0,
                    lambda_seed=lambda_seed, lambda_oov=lambda_oov,
                    caps_threshold=caps_threshold, role_mode=role_mode)

    if not role_caches_complete(data_dir, rp, cfg.data.dataset, cfg.data.split,
                                include_source=proto_roles is not None):
        log.info("annotation caches incomplete — loading Brown corpus + dataset stats")
        brown_freq, max_brown_freq = build_brown_freq() if role_mode in ("frequency", "variance") else ({}, 1.0)
        global_rate, global_count, dataset_presence, cross_dataset_rates = None, None, {}, {}
        if role_mode == "frequency":
            global_rate, global_count, dataset_presence = build_corpus_stats(data_dir)
        elif role_mode == "presence":
            _, _, dataset_presence = build_corpus_stats(data_dir)
        elif role_mode == "variance":
            cross_dataset_rates = build_cross_dataset_entity_rates(data_dir)
        rp = RoleParams(brown_freq=brown_freq, max_brown_freq=max_brown_freq,
                        lambda_seed=lambda_seed, lambda_oov=lambda_oov,
                        caps_threshold=caps_threshold, role_mode=role_mode,
                        dataset_presence=dataset_presence,
                        cross_dataset_rates=cross_dataset_rates)
    else:
        global_rate, global_count = None, None

    train_subset = OmegaConf.select(cfg, "data.train_subset")
    source = load_split(data_dir, cfg.data.dataset, cfg.data.split, subset=train_subset)
    entity_rate, source_count = compute_entity_rates(source)

    if proto_roles is not None:
        source = load_or_annotate_split(
            data_dir, cfg.data.dataset, cfg.data.split, cfg.data.dataset,
            entity_rate, source_count, rp, num_proc=num_workers or 1,
        )

    def _eval_nuc(nuc_pipeline):
        evaluate(nuc_pipeline, cfg, data_dir, entity_rate, source_count, rp,
                 global_rate=global_rate, global_count=global_count)

    pipeline = setup(cfg, source, device=device, num_workers=num_workers, batch_size=batch_size,
                     on_nuc_ready=_eval_nuc, proto_roles=proto_roles)
    del source

    metrics = evaluate(pipeline, cfg, data_dir, entity_rate, source_count, rp,
                       global_rate=global_rate, global_count=global_count)
    run.finish(metrics=metrics)
