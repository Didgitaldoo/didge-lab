"""
Evolution runner: population, mutation/crossover, selection by loss.
"""

import logging
import multiprocessing
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from tqdm import tqdm

from didgelab.app import get_app
from .genome import Genome
from .loss import LossFunction
from .operators import (
    MutationOperator,
    CrossoverOperator,
    RandomCrossover,
    AverageCrossover,
    SimpleMutation,
    SingleMutation,
    PartAverageCrossover,
    PartSwapCrossover,
)


class Nuevolution:
    """
    Evolution runner: maintains a population, applies mutation/crossover, and selects by loss.

    Initial population is built from father_genome (cloned and randomized) or from a list of genomes.
    Each generation, generation_size offspring are created via chosen operators, losses computed
    in parallel, then population is merged and trimmed to population_size by best total loss.
    Publishes generation_started, generation_ended, evolution_ended, log_evolution_operations.
    """

    def __init__(self,
                 loss: LossFunction,
                 father_genome: Genome,
                 generation_size=5,
                 num_generations=10,
                 population_size=10,
                 max_n_threads=None,
                 evolution_parameters={
                     "mutation_rate": 0.5,
                     "gene_mutation_prob": 0.5,
                 },
                 evolution_operators=None,
                 evolution_operator_probs=None,
                 callback_generation_ended=None):

        if evolution_operators is None:
            evolution_operators = [
                RandomCrossover(),
                AverageCrossover(),
                SimpleMutation(),
                SingleMutation(),
                PartAverageCrossover(),
                PartSwapCrossover()
            ]

        self.loss = loss
        self.father_genome = father_genome
        self.generation_size = generation_size
        self.num_generations = num_generations
        self.population_size = population_size
        self.evolution_parameters = evolution_parameters
        self.evolution_operators = evolution_operators
        self.max_n_threads = max_n_threads
        self.callback_generation_ended = callback_generation_ended

        if evolution_operator_probs is None:
            evolution_operator_probs = [1/len(evolution_operators)] * len(evolution_operators)
        self.evolution_operator_probs = evolution_operator_probs

        self.i_generation = -1
        self.population = None

        self.recompute_losses = False

        self.continue_evolution = True

        get_app().register_service(self)

        def recompute_loss():
            self.recompute_losses = True

        get_app().subscribe("recompute_loss", recompute_loss)

        if isinstance(father_genome, Genome):
            genome_type = type(father_genome).__name__
        elif isinstance(father_genome, list):
            genome_type = type(father_genome[0]).__name__
        else:
            raise Exception()

        logging_infos = {
            "loss": type(loss).__name__,
            "father_genome": genome_type,
            "generation_size": generation_size,
            "num_generations": num_generations,
            "population_size": population_size,
            "evolution_parameters": evolution_parameters,
            "evolution_operators": [type(o).__name__ for o in evolution_operators],
            "evolution_operator_probs": evolution_operator_probs
        }
        logging_infos = sorted([f"{key}: {value}" for key, value in logging_infos.items()])
        logging_infos = "Initialize Nuevolution\n" + "\n".join(logging_infos)
        logging.info(logging_infos)

    def get_evolution_progress(self):
        """Return progress in [0, 1]: (current_generation + 1) / num_generations."""
        return (self.i_generation + 1) / self.num_generations

    def evolve(self):
        """Run evolution for num_generations; return the final population (sorted by total loss)."""
        # initialize
        num_workers = multiprocessing.cpu_count()
        if self.max_n_threads is not None:
            num_workers = np.min((self.max_n_threads, num_workers))

        logging.info(f"initialize threadpoolexecutor with {num_workers} workers")
        pool = ThreadPoolExecutor(max_workers=num_workers)

        if isinstance(self.father_genome, Genome):
            self.population = []
            for i in range(self.generation_size):
                mutant = self.father_genome.clone()
                mutant.randomize_genome()
                self.population.append(mutant)
        elif isinstance(self.father_genome, list):
            self.population = self.father_genome
        else:
            raise Exception()

        logging.info("compute initial generation")

        losses = list(tqdm(pool.map(self.loss.loss, self.population), total=len(self.population)))
        for i in range(len(losses)):
            self.population[i].loss = losses[i]
        self.population = sorted(self.population, key=lambda x: x.loss["total"])

        get_app().publish("log_evolution_operations", (self.i_generation, self.population, [], []))
        # evolve
        probs = []
        for i_generation in range(1, self.num_generations + 1):

            if not self.continue_evolution:
                break

            get_app().publish("generation_started", (self.i_generation, self.population))

            if len(probs) != len(self.population):
                # compute probabilities that an individual will be selected
                probs = (1 + np.arange(len(self.population))) / len(self.population)
                probs = np.exp(probs)
                probs = np.exp(probs)
                probs = np.flip(probs)
                probs /= probs.sum()

            self.i_generation = i_generation

            if self.recompute_losses:
                losses = list(pool.map(self.loss.loss, self.population))
                self.recompute_losses = False

            operations = np.random.choice(
                self.evolution_operators, size=self.generation_size, p=self.evolution_operator_probs
            )

            new_generation = []
            mutation_operations = []
            crossover_operations = []
            for operator in operations:

                if isinstance(operator, MutationOperator):
                    parent = np.random.choice(self.population, p=probs)
                    individual, operation_description = operator.apply(parent, self.evolution_parameters)
                    mutation_operations.append(operation_description)
                    new_generation.append(individual)
                elif isinstance(operator, CrossoverOperator):
                    parent1 = np.random.choice(self.population, p=probs)
                    parent2 = parent1
                    while parent1.id == parent2.id:
                        parent2 = np.random.choice(self.population, p=probs)
                    individual, operation_description = operator.apply(parent1, parent2, self.evolution_parameters)
                    crossover_operations.append(operation_description)
                    new_generation.append(individual)

            # compute losses
            losses = list(pool.map(self.loss.loss, new_generation))
            for i in range(len(losses)):
                new_generation[i].loss = losses[i]

            # collect logging data
            get_app().publish(
                "log_evolution_operations",
                (self.i_generation, new_generation, mutation_operations, crossover_operations)
            )

            self.population = self.population + new_generation
            self.population = sorted(self.population, key=lambda x: x.loss["total"])

            if len(self.population) > self.population_size:
                self.population = self.population[0:self.population_size]

            get_app().publish("generation_ended", (self.i_generation, self.population))

            if self.callback_generation_ended is not None:
                self.callback_generation_ended(self.i_generation, self.population)

        get_app().publish("evolution_ended", (self.population))
        return self.population
