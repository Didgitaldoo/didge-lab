"""
Mutation and crossover operators for evolution.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Tuple

from .genome import Genome


class MutationOperator(ABC):
    """Abstract operator that mutates a single genome to produce one offspring."""

    @abstractmethod
    def apply(self, genome: Genome, evolution_parameters: Dict) -> Tuple[Genome, Dict]:
        """Apply mutation; return (child_genome, operation_description_dict)."""
        pass

    def describe(self, father_genome, child_genome):
        """Return a dict describing the mutation (operation name, father_id, child_id)."""
        return {
            "operation": type(self).__name__,
            "father_id": father_genome.id,
            "child_id": child_genome.id
        }


class CrossoverOperator(ABC):
    """Abstract operator that combines two parent genomes to produce one offspring."""

    @abstractmethod
    def apply(self, genome1: Genome, genome2: Genome, evolution_parameters: Dict) -> Tuple[Genome, Dict]:
        """Apply crossover; return (child_genome, operation_description_dict)."""
        pass

    def describe(self, parent1_genome, parent2_genome, child_genome):
        """Return a dict describing the crossover (operation name, parent ids, child_id)."""
        return {
            "operation": type(self).__name__,
            "parent1_genome": parent1_genome.id,
            "parent2_genome": parent2_genome.id,
            "child_id": child_genome.id
        }


class SimpleMutation(MutationOperator):
    """Add uniform random noise to genes; clamp to [0, 1]. Uses mutation_rate and gene_mutation_prob."""

    def apply(self, genome: Genome, evolution_parameters: Dict) -> Tuple[Genome, Dict]:

        mr = evolution_parameters["mutation_rate"]
        if mr is None:
            mr = 0.5
        mp = evolution_parameters["gene_mutation_prob"]
        if mp is None:
            mp = 0.5

        mutation = np.random.uniform(low=-mr, high=mr, size=len(genome.genome))
        mutation *= (np.random.sample(size=len(mutation)) < mp).astype(int)
        mutation = genome.genome + mutation
        mutation[mutation < 0] = 0
        mutation[mutation > 1] = 1

        new_genome = genome.clone()
        new_genome.genome = mutation
        return new_genome, self.describe(genome, new_genome)


class RandomMutation(MutationOperator):
    """Replace the entire genome with random values in [0, 1]."""

    def apply(self, genome: Genome, evolution_parameters: Dict) -> Tuple[Genome, Dict]:
        new_genome = genome.clone()
        new_genome.genome = np.random.sample(len(genome.genome))
        return new_genome, self.describe(genome, new_genome)


class SingleMutation(MutationOperator):
    """Change one randomly chosen gene by adding uniform noise in [-1, 1], then clamp to [0, 1]."""

    def apply(self, genome: Genome, evolution_parameters: Dict) -> Tuple[Genome, Dict]:
        new_genome = genome.clone()

        i = np.random.randint(0, len(new_genome.genome))
        v = new_genome.genome[i]
        v += np.random.uniform(-1, 1)
        if v < 0:
            v = 0
        if v > 1:
            v = 1
        new_genome.genome[i] = v
        return new_genome, self.describe(genome, new_genome)


class RandomCrossover(CrossoverOperator):
    """For each gene, choose randomly from parent1 or parent2."""

    def apply(self, parent1: Genome, parent2: Genome, evolution_parameters: Dict) -> Tuple[Genome, Dict]:
        assert type(parent1) == type(parent2)
        new_genome = list(zip(parent1.genome, parent2.genome))
        new_genome = np.array([np.random.choice(x) for x in new_genome])

        offspring = parent1.clone()
        offspring.genome = new_genome
        return offspring, self.describe(parent1, parent2, offspring)


class AverageCrossover(CrossoverOperator):
    """Each gene is the mean of parent1 and parent2."""

    def apply(self, parent1: Genome, parent2: Genome, evolution_parameters: Dict) -> Tuple[Genome, Dict]:
        assert type(parent1) == type(parent2)
        new_genome = (parent1.genome + parent2.genome) / 2

        offspring = parent1.clone()
        offspring.genome = new_genome
        return offspring, self.describe(parent1, parent2, offspring)


class PartSwapCrossover(CrossoverOperator):
    """Swap a contiguous slice of genes: parent1[0:i1], parent2[i1:i2], parent1[i2:]. Indices random."""

    def apply(self, parent1: Genome, parent2: Genome, evolution_parameters: Dict = {}) -> Tuple[Genome, Dict]:
        assert type(parent1) == type(parent2)

        indizes = np.random.choice(np.arange(len(parent1.genome)), size=2, replace=False)
        i1 = np.min(indizes)
        i2 = np.max(indizes)

        offspring = parent1.clone()
        offspring.genome = np.concatenate(
            (parent1.genome[0:i1],
             parent2.genome[i1:i2],
             parent1.genome[i2:]))

        assert len(offspring.genome) == len(parent1.genome)
        return offspring, self.describe(parent1, parent2, offspring)


class PartAverageCrossover(CrossoverOperator):
    """Average a contiguous slice of genes between parents; rest from parent1. Indices random."""

    def apply(self, parent1: Genome, parent2: Genome, evolution_parameters: Dict = {}) -> Tuple[Genome, Dict]:
        assert type(parent1) == type(parent2)

        indizes = np.random.choice(np.arange(len(parent1.genome)), size=2, replace=False)
        i1 = np.min(indizes)
        i2 = np.max(indizes)

        offspring = parent1.clone()
        offspring.genome = np.concatenate(
            (parent1.genome[0:i1],
             (parent1.genome[i1:i2] + parent2.genome[i1:i2]) / 2,
             parent1.genome[i2:]))

        assert len(offspring.genome) == len(parent1.genome)
        return offspring, self.describe(parent1, parent2, offspring)
