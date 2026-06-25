"""Main pipeline entry point: fit all active backends then evaluate.

Usage:
    forge run
    forge run nucleation.name=prototype
    forge run crystallization.name=boundary
    forge run nucleation.name=null crystallization.name=boundary
    forge run end2end.name=seq_labeler
    forge run data.train_subset=0.1 data.eval_subset=500
"""

from omegaconf import DictConfig
from pipeline.runner import run_pipeline


def main(cfg: DictConfig) -> None:
    run_pipeline(cfg)
