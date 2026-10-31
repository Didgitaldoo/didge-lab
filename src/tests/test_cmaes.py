"""
Pytest unit tests for didgelab.evo.cmaes (CMAES) and didgelab.evo.local_search (refine).
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

pytest.importorskip("configargparse")

from didgelab.evo import CMAES, Genome, LossFunction, refine, refine_population
from didgelab.evo.cmaes import reflect_into_unit_box


class DummyGenome(Genome):
    def genome2geo(self):
        return None


TARGET = np.linspace(0.2, 0.8, 10)


class QuadraticLoss(LossFunction):
    """Shifted sphere inside the unit box; counts evaluations."""

    def __init__(self, target=TARGET):
        self.target = target
        self.n_calls = 0

    def loss(self, genome):
        if genome.loss is not None:
            return genome.loss
        self.n_calls += 1
        return {"total": float(np.sum((genome.genome - self.target) ** 2))}


@pytest.fixture
def mock_app():
    app = MagicMock()
    with patch("didgelab.evo.cmaes.get_app", return_value=app):
        yield app


def test_reflect_into_unit_box():
    x = np.array([-0.25, 0.0, 0.5, 1.0, 1.25, 2.5, -1.5])
    np.testing.assert_allclose(reflect_into_unit_box(x), [0.25, 0.0, 0.5, 1.0, 0.75, 0.5, 0.5])


class TestCMAES:
    def test_converges_on_quadratic(self, mock_app):
        np.random.seed(0)
        loss = QuadraticLoss()
        evo = CMAES(loss, DummyGenome(n_genes=10), num_generations=150, keep=5)
        population = evo.evolve()

        assert len(population) == 5
        assert population == sorted(population, key=lambda x: x.loss["total"])
        assert population[0].loss["total"] < 1e-8
        assert evo.n_evaluations == loss.n_calls == 150 * evo.population_size

    def test_optimum_on_box_boundary(self, mock_app):
        np.random.seed(1)
        loss = QuadraticLoss(target=np.ones(10))
        population = CMAES(loss, DummyGenome(n_genes=10), num_generations=150).evolve()
        assert np.all((population[0].genome >= 0) & (population[0].genome <= 1))
        assert population[0].loss["total"] < 1e-8

    def test_start_from_father_with_small_sigma_stays_local(self, mock_app):
        np.random.seed(2)
        father = DummyGenome(genome=TARGET + 0.01)
        evo = CMAES(QuadraticLoss(), father, num_generations=1, sigma0=0.001, start_from_father=True)
        population = evo.evolve()
        assert np.max(np.abs(population[0].genome - father.genome)) < 0.01

    def test_continue_evolution_false_stops(self, mock_app):
        loss = QuadraticLoss()
        evo = CMAES(loss, DummyGenome(n_genes=10), num_generations=50, population_size=8)

        def stop_after_three(i_generation, population):
            if i_generation == 2:
                evo.continue_evolution = False

        evo.callback_generation_ended = stop_after_three
        evo.evolve()
        assert loss.n_calls == 3 * 8

    def test_recompute_loss_reevaluates_archive(self, mock_app):
        evo = CMAES(QuadraticLoss(), DummyGenome(n_genes=10), num_generations=3, population_size=8, keep=4)

        def swap_loss(i_generation, population):
            if i_generation == 1:
                evo.loss = QuadraticLoss(target=np.zeros(10))
                evo.recompute_losses = True

        evo.callback_generation_ended = swap_loss
        population = evo.evolve()
        for individual in population:
            assert individual.loss["total"] == pytest.approx(np.sum(individual.genome ** 2))


class TestRefine:
    @pytest.mark.parametrize("method", ["Powell", "Nelder-Mead"])
    def test_improves_and_respects_budget(self, method):
        loss = QuadraticLoss()
        start = DummyGenome(genome=np.full(10, 0.5))
        start.loss = loss.loss(start)

        result = refine(loss, start, max_evaluations=300, method=method)

        assert result.loss["total"] < 0.1 * start.loss["total"]
        assert np.all((result.genome >= 0) & (result.genome <= 1))
        assert result.named_parameters["refine_evaluations"] <= 300
        np.testing.assert_array_equal(start.genome, np.full(10, 0.5))

    def test_refine_population_replaces_starts(self):
        np.random.seed(3)
        loss = QuadraticLoss()
        population = []
        for _ in range(6):
            g = DummyGenome(n_genes=10)
            g.loss = loss.loss(g)
            population.append(g)
        best_before = min(p.loss["total"] for p in population)

        result = refine_population(loss, population, n_best=2, max_evaluations=200)

        assert len(result) == len(population)
        assert result == sorted(result, key=lambda x: x.loss["total"])
        assert result[0].loss["total"] < best_before
