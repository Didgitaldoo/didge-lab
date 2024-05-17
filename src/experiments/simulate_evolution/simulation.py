"""
python -m experiments.simulate_evolution.simulation
"""

from didgelab.evo.nuevolution import Nuevolution, NuevolutionWriter, \
    LossFunction, GeoGenomeA, LinearDecreasingCrossover, \
    LinearDecreasingMutation, AdaptiveProbabilities, Genome, \
    NuevolutionProgressBar, MutationOperator, RandomCrossover, \
    AverageCrossover, SimpleMutation, CrossoverOperator

import numpy as np
from typing import Dict, Tuple

class ConeLoss(LossFunction):

    def __init__(self):
        self.dstart = 32
        self.dend = 64

    def loss(self, shape : Genome):
        geo = shape.genome2geo().geo
        a = (self.dend-self.dstart)/(geo[-1][0])
        target = np.array([a*s[0] + self.dstart for s in geo])
        ys = np.array([s[1] for s in geo])
        loss = np.abs(target-ys).sum() / len(geo)
        return {"total": loss}


class TestGenome(GeoGenomeA):

    def build(n_segments):
        return TestGenome(n_genes=(n_segments*2)+1)

    def representation(self):
        geo = self.genome2geo()
        return {
            "geo": geo.geo,
        }

class SingleMutation(MutationOperator):

    def apply(self, genome : Genome, evolution_parameters : Dict) -> Tuple[Genome, Dict]:
        new_genome = genome.clone()

        i = np.random.randint(0, len(new_genome.genome))
        v = new_genome.genome[i]
        v += np.random.uniform(-1, 1)
        if v<0:
            v=0
        if v>1:
            v=1
        new_genome.genome[i] = v
        return new_genome, self.describe(genome, new_genome)
    
class PartSwapCrossover(CrossoverOperator):

    def apply(self, parent1 : Genome, parent2 : Genome, evolution_parameters : Dict = {}) -> Genome:
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

    def apply(self, parent1 : Genome, parent2 : Genome, evolution_parameters : Dict = {}) -> Genome:
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

def evolve():
    writer = NuevolutionWriter(write_population_interval=50)

    evo = Nuevolution(
        ConeLoss(), 
        TestGenome.build(10),
        num_generations=100,
        population_size=1000,
        generation_size=50,
        evolution_operators = [
            SingleMutation(), 
            RandomCrossover(), 
            AverageCrossover(), 
            SimpleMutation(), 
            PartAverageCrossover(),
            PartSwapCrossover()]
        )

    NuevolutionProgressBar()
    schedulers = [
        LinearDecreasingMutation()
    ]
    AdaptiveProbabilities()
    evo.evolve()

evolve()