"""Standalone evaluation entry point (re-fits backends from source data).

Will diverge from train.py once backends support serialisation.

Usage:
    forge -M evaluate run
    forge -M evaluate run data.dataset=wikiann data.split=train
    forge -M evaluate run data.eval_subset=200
"""

from omegaconf import DictConfig
from pipeline.runner import run_pipeline


def main(cfg: DictConfig) -> None:
    run_pipeline(cfg)
