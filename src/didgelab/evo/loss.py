"""
Loss functions for evaluating genomes. Lower loss is better.
"""

import numpy as np
from abc import ABC, abstractmethod

from .genome import Genome


class LossFunction(ABC):
    """Abstract loss used to evaluate a genome. Lower loss is better."""

    @abstractmethod
    def loss(self, shape: Genome):
        """Compute loss for the given genome. Returns a dict (must include 'total' for selection)."""
        pass


class TestLossFunction(LossFunction):
    """Example loss for testing: sum(first half) / sum(second half) plus dummy keys."""

    def loss(self, genome: Genome):
        l = int(len(genome.genome)/2)
        return {"total": np.sum(genome.genome[0:l]) / np.sum(genome.genome[l:]), "test": -5, "test2": 10}
