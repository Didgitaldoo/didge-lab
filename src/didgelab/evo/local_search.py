"""
Local refinement of evolved genomes (hybrid search).

A global search (Nuevolution or CMAES) finds good regions; a local optimizer then polishes
the best genomes. Two kinds:

* ``refine`` with "Powell" / "Nelder-Mead": derivative-free, works with any loss.
* ``refine_least_squares``: Gauss-Newton style least squares on the signed residuals of a
  CompositeTairuaLoss, with a finite-difference Jacobian. Much stronger for tuning targets
  (FrequencyTuningLoss, ScaleTuningLoss); see doc/examples/least_squares_refinement.ipynb.
"""

import multiprocessing
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import List

import numpy as np
from scipy.optimize import least_squares, minimize

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
        method: "Powell" (coordinate line searches, good in many dimensions),
            "Nelder-Mead" (simplex of size ``initial_step``) or "least-squares"
            (``refine_least_squares``; needs a loss with ``residuals``).
        initial_step: Nelder-Mead simplex edge length in genome units.

    Returns:
        The best genome evaluated (a clone with ``.loss`` set); the start genome itself
        if nothing better was found. ``.named_parameters["refine_evaluations"]`` holds the
        number of loss evaluations used.
    """
    if method == "least-squares":
        return refine_least_squares(loss, genome, max_evaluations)

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


class _BudgetExhausted(Exception):
    pass


def refine_least_squares(loss, genome: Genome, max_evaluations: int = 500,
                         fd_step: float = 1e-6, max_n_threads: int = None) -> Genome:
    """
    Push ``genome`` to the nearest local minimum by least squares on the loss residuals.

    The loss of a tuning target is a sum of absolute deviations, which has a kink exactly
    at the target. This minimizes the sum of the squared signed deviations instead
    (``loss.residuals``), which is smooth there, with scipy's trust-region-reflective
    least squares inside the [0, 1] box. The Jacobian comes from forward finite differences
    (n_genes + 1 evaluations, run in parallel threads); the loss is smooth enough at the
    scale of ``fd_step`` for them to be accurate.

    Works best when the loss is dominated by FrequencyTuningLoss and ScaleTuningLoss.
    Components without residuals enter as sqrt(value) and keep their kinks.

    Args:
        loss: A loss with ``residuals(shape) -> (vector or None, results)`` and ``loss``,
            e.g. CompositeTairuaLoss.
        genome: Start point, typically the best genome of an evolution; not modified.
        max_evaluations: Budget of loss evaluations (one Jacobian costs n_genes + 1).
        fd_step: Finite-difference step in genome units.
        max_n_threads: Threads for the Jacobian (default: cpu count).

    Returns:
        The genome with the lowest total loss evaluated (a clone with ``.loss`` set), at
        worst a clone of the start genome. ``.named_parameters
        ["refine_evaluations"]`` holds the number of loss evaluations used.
    """
    if not hasattr(loss, "residuals"):
        raise TypeError(f"{type(loss).__name__} has no residuals(); use refine() with Powell instead")

    lock = threading.Lock()
    evaluations = [0]
    best = [None]

    def evaluate(x):
        with lock:
            if evaluations[0] >= max_evaluations:
                raise _BudgetExhausted()
            evaluations[0] += 1
        candidate = genome.clone()
        candidate.genome = np.clip(x, 0.0, 1.0)
        try:
            residuals, results = loss.residuals(candidate)
        except Exception:
            residuals, results = None, {"total": np.inf}
        candidate.loss = results
        with lock:
            if best[0] is None or results["total"] < best[0].loss["total"]:
                best[0] = candidate
        return residuals

    x0 = np.clip(np.asarray(genome.genome, dtype=float), 0.0, 1.0)
    r0 = evaluate(x0)
    if r0 is None or len(r0) == 0:
        return _with_evaluations(best[0], evaluations[0])
    n_residuals = len(r0)
    failed = np.full(n_residuals, 1e3)

    def residual_fn(x):
        # Fixed length: the number of peaks (and so of scale residuals) can change
        r = evaluate(x)
        if r is None:
            return failed
        if len(r) >= n_residuals:
            return r[:n_residuals]
        return np.concatenate([r, np.zeros(n_residuals - len(r))])

    n = len(x0)
    basis = np.eye(n)
    pool = ThreadPoolExecutor(max_workers=max_n_threads or multiprocessing.cpu_count())

    def jacobian(x):
        r = residual_fn(x)
        steps = np.where(x + fd_step <= 1.0, fd_step, -fd_step)  # step inwards at the upper bound
        shifted = list(pool.map(lambda i: residual_fn(x + steps[i] * basis[i]), range(n)))
        return np.column_stack([(s - r) / h for s, h in zip(shifted, steps)])

    try:
        least_squares(residual_fn, x0, jac=jacobian, bounds=(0.0, 1.0), method="trf",
                      x_scale="jac", ftol=1e-12, xtol=1e-12, gtol=1e-12, max_nfev=10 ** 6)
    except _BudgetExhausted:
        pass
    finally:
        pool.shutdown()

    # The start point was the first evaluation, so the result is never worse than it
    return _with_evaluations(best[0], evaluations[0])


def _with_evaluations(genome: Genome, evaluations: int) -> Genome:
    genome.named_parameters = dict(genome.named_parameters, refine_evaluations=evaluations)
    return genome
