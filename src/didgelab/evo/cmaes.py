"""
CMA-ES runner: covariance matrix adaptation evolution strategy on the [0, 1]^n genome.

Same loss interface and evolve() contract as Nuevolution, but instead of mutation and
crossover operators it samples each generation from a multivariate normal distribution
whose mean, step size and covariance are adapted from the best samples. On smooth
continuous genomes like the bore shapes this usually needs far fewer loss evaluations.

Follows the standard (mu/mu_w, lambda)-CMA-ES of Hansen, "The CMA Evolution Strategy:
A Tutorial" (arXiv:1604.00772). Samples outside [0, 1] are reflected back into the box
before evaluation and the reflected point is used for the update.
"""

import logging
import multiprocessing
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from didgelab.app import get_app
from .genome import Genome
from .loss import LossFunction


def reflect_into_unit_box(x: np.ndarray) -> np.ndarray:
    """Mirror values at 0 and 1 until they lie in [0, 1] (period 2 reflection)."""
    y = np.mod(x, 2.0)
    return np.where(y > 1.0, 2.0 - y, y)


class CMAES:
    """
    CMA-ES runner with the Nuevolution evolve() contract.

    Each generation samples ``population_size`` genomes, evaluates their losses in parallel
    and adapts the search distribution. ``evolve()`` returns the ``keep`` best genomes seen
    over the whole run, sorted by total loss. Publishes generation_started, generation_ended
    and evolution_ended; honours ``continue_evolution`` and the ``recompute_loss`` event.

    Callbacks that look up the Nuevolution service (schedulers, AdaptiveProbabilities,
    NuevolutionWriter) do not apply here: CMA-ES adapts its own step size and has no operators.
    """

    def __init__(self,
                 loss: LossFunction,
                 father_genome: Genome,
                 num_generations: int = 100,
                 population_size: int = None,
                 sigma0: float = 0.3,
                 keep: int = 10,
                 start_from_father: bool = False,
                 max_n_threads: int = None,
                 callback_generation_ended=None,
                 callback_loss_progress=None):
        """
        Args:
            loss: Loss function; ``loss.loss(genome)`` returns a dict with "total".
            father_genome: Template genome; cloned for every sample.
            num_generations: Number of generations (population_size evaluations each).
            population_size: Samples per generation (lambda). Default 4 + 3 ln(n_genes).
            sigma0: Initial step size in genome units (the box is [0, 1]).
            keep: Number of best genomes returned by evolve().
            start_from_father: Start the mean at father_genome.genome instead of the box
                centre. Use with a small sigma0 to refine an existing shape locally.
            max_n_threads: Upper bound on parallel loss evaluations (default: cpu count).
        """
        self.loss = loss
        self.father_genome = father_genome
        self.num_generations = num_generations
        self.n = len(father_genome.genome)
        self.population_size = population_size or 4 + int(3 * np.log(self.n))
        self.sigma0 = sigma0
        self.keep = keep
        self.start_from_father = start_from_father
        self.max_n_threads = max_n_threads
        self.callback_generation_ended = callback_generation_ended
        self.callback_loss_progress = callback_loss_progress

        self.i_generation = -1
        self.population = []
        self.n_evaluations = 0
        self.continue_evolution = True
        self.recompute_losses = False

        get_app().register_service(self)

        def recompute_loss():
            self.recompute_losses = True

        get_app().subscribe("recompute_loss", recompute_loss)

        logging.info(
            "Initialize CMAES: loss %s, genome %s (%d genes), lambda %d, sigma0 %.3f, %d generations",
            type(loss).__name__, type(father_genome).__name__, self.n,
            self.population_size, sigma0, num_generations,
        )

    def get_evolution_progress(self):
        """Return progress in [0, 1]: (current_generation + 1) / num_generations."""
        return (self.i_generation + 1) / self.num_generations

    def _evaluate(self, pool, individuals):
        """Compute losses in parallel, reporting progress as each completes."""
        results = []
        for i, loss in enumerate(pool.map(self.loss.loss, individuals)):
            results.append(loss)
            if self.callback_loss_progress is not None:
                self.callback_loss_progress(self.i_generation, i + 1, len(individuals))
        for individual, loss in zip(individuals, results):
            individual.loss = loss
        self.n_evaluations += len(individuals)

    def _update_archive(self, individuals):
        """Keep the `keep` best genomes seen so far, sorted by total loss."""
        merged = self.population + list(individuals)
        self.population = sorted(merged, key=lambda x: x.loss["total"])[:self.keep]

    def evolve(self):
        """Run CMA-ES for num_generations; return the best genomes found (sorted by total loss)."""
        n = self.n
        lam = self.population_size
        mu = lam // 2

        # Recombination weights and strategy parameters (Hansen's defaults)
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1.0 / np.sum(weights ** 2)
        cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
        cs = (mueff + 2) / (n + mueff + 5)
        c1 = 2 / ((n + 1.3) ** 2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))
        damps = 1 + 2 * max(0.0, np.sqrt((mueff - 1) / (n + 1)) - 1) + cs
        chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

        mean = np.array(self.father_genome.genome, dtype=float) if self.start_from_father else np.full(n, 0.5)
        sigma = self.sigma0
        C = np.eye(n)
        B = np.eye(n)
        D = np.ones(n)
        pc = np.zeros(n)
        ps = np.zeros(n)

        num_workers = multiprocessing.cpu_count()
        if self.max_n_threads is not None:
            num_workers = min(self.max_n_threads, num_workers)
        pool = ThreadPoolExecutor(max_workers=num_workers)

        for i_generation in range(self.num_generations):
            if not self.continue_evolution:
                break
            self.i_generation = i_generation
            get_app().publish("generation_started", (self.i_generation, self.population))

            if self.recompute_losses:
                for individual in self.population:
                    individual.loss = None
                self._evaluate(pool, self.population)
                self.population = sorted(self.population, key=lambda x: x.loss["total"])
                self.recompute_losses = False

            # Sample lambda offspring: x = mean + sigma * B D z, reflected into [0, 1]
            z = np.random.standard_normal((lam, n))
            x = reflect_into_unit_box(mean + sigma * (z * D) @ B.T)
            offspring = []
            for genes in x:
                individual = self.father_genome.clone()
                individual.genome = genes
                offspring.append(individual)
            self._evaluate(pool, offspring)
            self._update_archive(offspring)

            # Update mean from the mu best (repaired) samples
            order = np.argsort([o.loss["total"] for o in offspring])
            x_sel = x[order[:mu]]
            mean_old = mean
            mean = weights @ x_sel
            y_w = (mean - mean_old) / sigma

            # Step-size and covariance evolution paths
            inv_sqrt_C_y = B @ ((B.T @ y_w) / D)
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * inv_sqrt_C_y
            ps_norm = np.linalg.norm(ps)
            hsig = ps_norm / np.sqrt(1 - (1 - cs) ** (2 * (i_generation + 1))) / chi_n < 1.4 + 2 / (n + 1)
            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * y_w

            y_sel = (x_sel - mean_old) / sigma
            C = ((1 - c1 - cmu) * C
                 + c1 * (np.outer(pc, pc) + (1 - hsig) * cc * (2 - cc) * C)
                 + cmu * (y_sel.T * weights) @ y_sel)
            sigma *= np.exp((cs / damps) * (ps_norm / chi_n - 1))
            sigma = min(sigma, 1.0)

            # Decompose C = B diag(D^2) B^T
            C = np.triu(C) + np.triu(C, 1).T
            eigenvalues, B = np.linalg.eigh(C)
            D = np.sqrt(np.maximum(eigenvalues, 1e-20))

            self.sigma = sigma
            self.mean = mean
            get_app().publish("generation_ended", (self.i_generation, self.population))
            if self.callback_generation_ended is not None:
                self.callback_generation_ended(self.i_generation, self.population)

            if sigma * D.max() < 1e-8:
                logging.info("CMAES converged at generation %d", i_generation)
                break

        pool.shutdown()
        get_app().publish("evolution_ended", (self.population))
        return self.population
