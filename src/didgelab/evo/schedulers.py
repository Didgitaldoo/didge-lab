"""
Schedulers for mutation/crossover rates and adaptive operator probabilities.
"""

import numpy as np

from didgelab.app import get_app
from .evolution import Nuevolution


class LinearDecreasingMutation:
    """Scheduler that decreases mutation_rate and gene_mutation_prob over evolution progress (4 steps)."""

    def __init__(self):

        n_steps = 4
        self.schedule = np.arange(1, n_steps + 1) / n_steps
        self.rates = 1 - (np.arange(n_steps) / n_steps)
        self.i = -1

        def update(i_generation, population):

            if self.i >= len(self.schedule):
                return

            nuevolution = get_app().get_service(Nuevolution)
            progress = nuevolution.get_evolution_progress()
            if self.i == -1 or progress > self.schedule[self.i]:
                self.i += 1
                rate = self.rates[self.i]
                nuevolution.evolution_parameters["mutation_rate"] = rate
                nuevolution.evolution_parameters["gene_mutation_prob"] = rate
                get_app().publish("recompute_loss")

        get_app().subscribe("generation_started", update)


class LinearDecreasingCrossover:
    """Scheduler that decreases crossover_prob in evolution_parameters over progress (4 steps)."""

    def __init__(self):

        n_steps = 4
        self.schedule = np.arange(1, n_steps + 1) / n_steps
        self.rates = 0.5 - (np.arange(n_steps) / n_steps)
        self.rates = [np.max(x, 0) for x in self.rates]
        self.i = -1

        def update(i_generation, population):
            if self.i >= len(self.schedule):
                return

            nuevolution = get_app().get_service(Nuevolution)
            progress = nuevolution.get_evolution_progress()
            if self.i == -1 or progress > self.schedule[self.i]:
                self.i += 1
                rate = self.rates[self.i]
                nuevolution.evolution_parameters["crossover_prob"] = rate
                get_app().publish("recompute_loss")
        get_app().subscribe("generation_started", update)


class AdaptiveProbabilities:
    """
    Adjusts evolution operator probabilities from recent loss deltas (success = child better than parent).

    Subscribes to log_evolution_operations; keeps a sliding window of deltas per operator and sets
    probabilities proportional to success rate (min 0.05), then updates Nuevolution.evolution_operator_probs.
    """

    def __init__(self):
        get_app().subscribe("log_evolution_operations", self.log_evolution_operations)
        self.loss_index = {}
        self.loss_history = {}
        self.window_size = 100
        self.probabilities = {}

    def compute_loss_delta_of_generation(self, mutation_operations, crossover_operations):
        """Return a dict mapping each operation name to a list of (child_loss - best_parent_loss) deltas."""
        generation = {}
        for mutation in mutation_operations:

            if mutation["father_id"] not in self.loss_index:
                continue
            if mutation["child_id"] not in self.loss_index:
                continue

            father_loss = self.loss_index[mutation["father_id"]]
            child_loss = self.loss_index[mutation["child_id"]]

            if mutation["operation"] not in generation:
                generation[mutation["operation"]] = []

            generation[mutation["operation"]].append(child_loss - father_loss)

        for crossover in crossover_operations:
            if crossover["parent1_genome"] not in self.loss_index:
                continue
            if crossover["parent2_genome"] not in self.loss_index:
                continue
            if crossover["child_id"] not in self.loss_index:
                continue

            p1_loss = self.loss_index[crossover["parent1_genome"]]
            p2_loss = self.loss_index[crossover["parent2_genome"]]
            child_loss = self.loss_index[crossover["child_id"]]

            minloss = np.min((p1_loss, p2_loss))
            if crossover["operation"] not in generation:
                generation[crossover["operation"]] = []
            generation[crossover["operation"]].append(child_loss - minloss)

        return generation

    def log_evolution_operations(self, i_generation, individuals, mutation_operations, crossover_operations):
        """Update loss index, compute deltas for this generation, and update operator probabilities."""
        if len(self.loss_index) == 0:
            pop = get_app().get_service(Nuevolution).population
            for i in pop:
                self.loss_index[i.id] = i.loss["total"]

        for i in range(len(individuals)):
            self.loss_index[individuals[i].id] = individuals[i].loss["total"]

        generation = self.compute_loss_delta_of_generation(mutation_operations, crossover_operations)

        for operation, losses in generation.items():
            if operation not in self.loss_history:
                self.loss_history[operation] = []
            self.loss_history[operation].extend(losses)
            if len(self.loss_history[operation]) > self.window_size:
                self.loss_history[operation] = self.loss_history[operation][-self.window_size:]

            num_successful = (np.array(self.loss_history[operation]) < 0).astype(int).sum()
            num_total = len(self.loss_history[operation])
            self.probabilities[operation] = np.max((num_successful / num_total, 0.05))

        # normalize probabilities
        prob_sum = np.sum(list(self.probabilities.values()))
        self.probabilities = {key: value / prob_sum for key, value in self.probabilities.items()}

        # update probs in nuevolution
        nuevo = get_app().get_service(Nuevolution)
        probs = []
        for i in range(len(nuevo.evolution_operators)):
            o = nuevo.evolution_operators[i]
            if type(o).__name__ in self.probabilities:
                p = self.probabilities[type(o).__name__]
            else:
                p = nuevo.evolution_operator_probs[i]
            probs.append(p)
        nuevo.evolution_operator_probs = probs
