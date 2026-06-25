"""Nucleation-only evaluation (oracle crystallization).

Equivalent to: forge run crystallization.name=null

Usage:
    forge -M nucleation run
    forge -M nucleation run data.dataset=wikiann
"""

from omegaconf import DictConfig
from pipeline.runner import run_pipeline


def main(cfg: DictConfig) -> None:
    run_pipeline(cfg, force_cry=None)
