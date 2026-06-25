"""Crystallization-only evaluation (oracle nucleation).

Equivalent to: forge run nucleation.name=null

Usage:
    forge -M crystallization run
    forge -M crystallization run crystallization.name=smoothing
"""

from omegaconf import DictConfig
from pipeline.runner import run_pipeline


def main(cfg: DictConfig) -> None:
    run_pipeline(cfg, force_nuc=None)
