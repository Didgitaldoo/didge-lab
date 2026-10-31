"""
Local refinement of evolved genomes (hybrid search).

A global search (Nuevolution or CMAES) finds good regions; a derivative-free local
optimizer then polishes the best genomes. Useful because the loss is smooth near good
shapes now that peak frequencies are interpolated instead of grid-quantized.
"""

from concurrent.futures import ThreadPoolExecutor
from typing import List

import numpy as np
from scipy.optimize import minimize

from .genome import Genome
from .loss import LossFunction


def refine(loss: LossFunction, genome: Genome, max_evaluations: int = 500,
           method: str = "Powell", initial_step: float = 0.05) -> Genome:
    """
    Locally minimize the total loss starting from ``genome``, within the [0, 1] box.

    Args:
        loss: Loss function; ``loss.loss(genome)`` returns a dict with "total".
        genome: Start point; not modified.
        max_evaluations: Budget of loss evaluations.
        method: "Powell" (coordinate line searches, good in many dimensions) or
            "Nelder-Mead" (simplex of size ``initial_step``).
        initial_step: Nelder-Mead simplex edge length in genome units.

    Returns:
        The best genome evaluated (a clone with ``.loss`` set); the start genome itself
        if nothing better was found. ``.named_parameters["refine_evaluations"]`` holds the
        number of loss evaluations used.
    """
    start = genome.clone()
    start.genome = np.clip(np.asarray(genome.genome, dtype=float), 0.0, 1.0)
    if genome.loss is not None:
        start.loss = genome.loss
    else:
        start.loss = loss.loss(start)
    best = [start]
    evaluations = [0]

    def objective(x):
        if evaluations[0] >= max_evaluations:
            return best[0].loss["total"]
        candidate = genome.clone()
        candidate.genome = np.clip(x, 0.0, 1.0)
        candidate.loss = loss.loss(candidate)
        evaluations[0] += 1
        if candidate.loss["total"] < best[0].loss["total"]:
            best[0] = candidate
        return candidate.loss["total"]

    x0 = start.genome
    n = len(x0)
    options = {"maxfev": max_evaluations}
    if method == "Nelder-Mead":
        # Step each coordinate inwards so the whole simplex lies in the box
        steps = np.where(x0 + initial_step <= 1.0, initial_step, -initial_step)
        options["initial_simplex"] = np.vstack([x0, x0 + np.diag(steps)])
        options["adaptive"] = n > 10
    minimize(objective, x0, method=method, bounds=[(0.0, 1.0)] * n, options=options)

    result = best[0]
    result.named_parameters = dict(result.named_parameters, refine_evaluations=evaluations[0])
    return result


def refine_population(loss: LossFunction, population: List[Genome], n_best: int = 4,
                      max_evaluations: int = 500, method: str = "Powell",
                      max_n_threads: int = None) -> List[Genome]:
    """
    Refine the ``n_best`` best genomes of ``population`` in parallel (one local search
    per thread). Each refined genome replaces its start genome (it is never worse);
    returns the population sorted by total loss.
    """
    ranked = sorted(population, key=lambda x: x.loss["total"])
    starts = ranked[:n_best]
    with ThreadPoolExecutor(max_workers=max_n_threads or len(starts) or 1) as pool:
        refined = list(pool.map(lambda g: refine(loss, g, max_evaluations, method), starts))
    return sorted(refined + ranked[n_best:], key=lambda x: x.loss["total"])
