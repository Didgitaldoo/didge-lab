"""
Pytest unit tests for didgelab.evo (evolutionary optimization).
"""

import json
import sys
import pytest
import numpy as np
from unittest.mock import MagicMock, patch

# evo imports didgelab.app (needs configargparse) and sim (needs _cadsd).
# Inject mock _cadsd before sim loads, so tests run without building the Cython extension.
pytest.importorskip("configargparse")

_mock_cadsd = MagicMock()

def _mock_create_segments_from_geo(geo):
    from didgelab.sim.tlm_python import Segment
    return Segment.create_segments_from_geo(geo)

def _mock_cadsd_Ze(segments, freq):
    return 1e6  # placeholder impedance magnitude

_mock_cadsd.create_segments_from_geo = _mock_create_segments_from_geo
_mock_cadsd.cadsd_Ze = _mock_cadsd_Ze
sys.modules["didgelab.sim.tlm_cython_lib._cadsd"] = _mock_cadsd

from didgelab.geo import Geo
from didgelab.evo import (
    Genome,
    GeoGenome,
    GeoGenomeA,
    LossFunction,
    MutationOperator,
    CrossoverOperator,
    SimpleMutation,
    RandomMutation,
    SingleMutation,
    RandomCrossover,
    AverageCrossover,
    PartSwapCrossover,
    PartAverageCrossover,
    TestLossFunction,
    NumpyEncoder,
    Nuevolution,
    AdaptiveProbabilities,
)


class DummyGenome(Genome):
    """Concrete Genome for testing base class behavior."""

    def genome2geo(self):
        return None


class TestGenome:
    """Tests for Genome base class."""

    def test_init_from_n_genes(self):
        g = DummyGenome(n_genes=10)
        assert len(g.genome) == 10
        assert np.all((g.genome >= 0) & (g.genome <= 1))
        assert g.loss is None
        assert g.id >= 0

    def test_init_from_genome(self):
        genome = np.array([0.1, 0.2, 0.3])
        g = DummyGenome(genome=genome)
        np.testing.assert_array_equal(g.genome, genome)

    def test_init_requires_n_genes_or_genome(self):
        with pytest.raises(AssertionError):
            DummyGenome()

    def test_generate_id_unique(self):
        ids = [Genome.generate_id() for _ in range(10)]
        assert len(set(ids)) == 10

    def test_representation(self):
        genome = np.array([0.1, 0.2, 0.3])
        g = DummyGenome(genome=genome)
        rep = g.representation()
        assert rep == [0.1, 0.2, 0.3]

    def test_randomize_genome(self):
        g = DummyGenome(n_genes=5)
        orig = g.genome.copy()
        g.randomize_genome()
        assert len(g.genome) == 5
        assert not np.allclose(g.genome, orig)

    def test_clone(self):
        g = DummyGenome(genome=np.array([0.1, 0.2]))
        g.loss = {"total": 1.0}
        c = g.clone()
        assert c.id != g.id
        assert c.loss is None
        np.testing.assert_array_equal(c.genome, g.genome)
        assert c is not g


class TestGeoGenomeA:
    """Tests for GeoGenomeA genome type."""

    def test_build(self):
        g = GeoGenomeA.build(3)
        assert isinstance(g, GeoGenomeA)
        assert len(g.genome) == 3 * 2 + 1  # n_segments*2 + 1

    def test_genome2geo_returns_geo(self):
        g = GeoGenomeA(n_genes=7)
        g.genome = np.array([0.5, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])  # reproducible
        geo = g.genome2geo()
        assert isinstance(geo, Geo)
        assert len(geo.geo) >= 2

    def test_genome2geo_mouth_fixed_at_32(self):
        g = GeoGenomeA(n_genes=5)
        geo = g.genome2geo()
        assert geo.geo[0][1] == 32

    def test_genome2geo_length_in_range(self):
        g = GeoGenomeA(n_genes=7)
        for _ in range(5):  # random init, check several
            g.randomize_genome()
            geo = g.genome2geo()
            length = geo.length()
            assert 1000 <= length <= 2000


class TestTestLossFunction:
    """Tests for TestLossFunction (example loss)."""

    def test_returns_dict_with_total(self):
        loss_fn = TestLossFunction()
        g = DummyGenome(genome=np.array([0.5, 0.5, 0.5, 0.5]))
        result = loss_fn.loss(g)
        assert "total" in result
        assert "test" in result
        assert "test2" in result

    def test_total_is_first_half_over_second_half(self):
        loss_fn = TestLossFunction()
        g = DummyGenome(genome=np.array([1.0, 1.0, 0.5, 0.5]))
        result = loss_fn.loss(g)
        expected = (1.0 + 1.0) / (0.5 + 0.5)
        assert result["total"] == pytest.approx(expected)


class TestSimpleMutation:
    """Tests for SimpleMutation operator."""

    def test_produces_valid_genome(self):
        op = SimpleMutation()
        g = DummyGenome(genome=np.array([0.5, 0.5, 0.5]))
        params = {"mutation_rate": 0.1, "gene_mutation_prob": 0.5}
        child, desc = op.apply(g, params)
        assert np.all((child.genome >= 0) & (child.genome <= 1))
        assert len(child.genome) == len(g.genome)
        assert desc["operation"] == "SimpleMutation"
        assert desc["child_id"] == child.id

    def test_uses_default_params_when_none(self):
        op = SimpleMutation()
        g = DummyGenome(genome=np.array([0.5, 0.5]))
        child, _ = op.apply(g, {"mutation_rate": None, "gene_mutation_prob": None})
        assert np.all((child.genome >= 0) & (child.genome <= 1))


class TestRandomMutation:
    """Tests for RandomMutation operator."""

    def test_replaces_genome(self):
        op = RandomMutation()
        g = DummyGenome(genome=np.array([0.1, 0.2, 0.3]))
        child, desc = op.apply(g, {})
        assert len(child.genome) == len(g.genome)
        assert np.all((child.genome >= 0) & (child.genome <= 1))
        assert desc["father_id"] == g.id
        assert desc["child_id"] == child.id


class TestSingleMutation:
    """Tests for SingleMutation operator."""

    def test_changes_one_gene(self):
        op = SingleMutation()
        g = DummyGenome(genome=np.array([0.5, 0.5, 0.5]))
        child, _ = op.apply(g, {})
        assert len(child.genome) == len(g.genome)
        assert np.all((child.genome >= 0) & (child.genome <= 1))
        # At least one gene may differ (stochastic)
        diff_count = np.sum(child.genome != g.genome)
        assert diff_count <= 1


class TestRandomCrossover:
    """Tests for RandomCrossover operator."""

    def test_offspring_from_both_parents(self):
        op = RandomCrossover()
        np.random.seed(42)
        p1 = DummyGenome(genome=np.array([0.0, 0.0, 0.0]))
        p2 = DummyGenome(genome=np.array([1.0, 1.0, 1.0]))
        child, desc = op.apply(p1, p2, {})
        assert len(child.genome) == len(p1.genome)
        assert np.all((child.genome >= 0) & (child.genome <= 1))
        assert desc["parent1_genome"] == p1.id
        assert desc["parent2_genome"] == p2.id

    def test_same_type_parents_required(self):
        op = RandomCrossover()
        p1 = DummyGenome(genome=np.array([0.5, 0.5]))
        p2 = DummyGenome(genome=np.array([0.5, 0.5]))
        child, _ = op.apply(p1, p2, {})
        assert type(child) == type(p1)


class TestAverageCrossover:
    """Tests for AverageCrossover operator."""

    def test_offspring_is_average(self):
        op = AverageCrossover()
        p1 = DummyGenome(genome=np.array([0.0, 1.0, 0.0]))
        p2 = DummyGenome(genome=np.array([1.0, 0.0, 1.0]))
        child, _ = op.apply(p1, p2, {})
        expected = np.array([0.5, 0.5, 0.5])
        np.testing.assert_array_almost_equal(child.genome, expected)


class TestPartSwapCrossover:
    """Tests for PartSwapCrossover operator."""

    def test_offspring_same_length(self):
        op = PartSwapCrossover()
        np.random.seed(123)
        p1 = DummyGenome(genome=np.array([0.1, 0.2, 0.3, 0.4, 0.5]))
        p2 = DummyGenome(genome=np.array([0.9, 0.8, 0.7, 0.6, 0.5]))
        child, _ = op.apply(p1, p2, {})
        assert len(child.genome) == len(p1.genome)

    def test_contains_genes_from_both(self):
        op = PartSwapCrossover()
        np.random.seed(456)
        p1 = DummyGenome(genome=np.arange(10, dtype=float) / 10)
        p2 = DummyGenome(genome=np.ones(10))
        child, _ = op.apply(p1, p2, {})
        assert len(child.genome) == 10


class TestPartAverageCrossover:
    """Tests for PartAverageCrossover operator."""

    def test_offspring_same_length(self):
        op = PartAverageCrossover()
        np.random.seed(789)
        p1 = DummyGenome(genome=np.array([0.1, 0.2, 0.3, 0.4, 0.5]))
        p2 = DummyGenome(genome=np.array([0.9, 0.8, 0.7, 0.6, 0.5]))
        child, _ = op.apply(p1, p2, {})
        assert len(child.genome) == len(p1.genome)


class TestNumpyEncoder:
    """Tests for NumpyEncoder JSON serialization."""

    def test_numpy_int(self):
        encoded = json.dumps({"x": np.int64(42)}, cls=NumpyEncoder)
        obj = json.loads(encoded)
        assert obj["x"] == 42
        assert isinstance(obj["x"], int)

    def test_numpy_float(self):
        encoded = json.dumps({"x": np.float64(3.14)}, cls=NumpyEncoder)
        obj = json.loads(encoded)
        assert obj["x"] == pytest.approx(3.14)
        assert isinstance(obj["x"], float)

    def test_numpy_array(self):
        encoded = json.dumps({"x": np.array([1, 2, 3])}, cls=NumpyEncoder)
        obj = json.loads(encoded)
        assert obj["x"] == [1, 2, 3]


class TestAdaptiveProbabilities:
    """Tests for AdaptiveProbabilities.compute_loss_delta_of_generation."""

    def test_compute_loss_delta_mutation(self):
        ap = AdaptiveProbabilities.__new__(AdaptiveProbabilities)
        ap.loss_index = {1: 10.0, 2: 8.0}
        mutation_ops = [{"father_id": 1, "child_id": 2, "operation": "SimpleMutation"}]
        crossover_ops = []
        result = ap.compute_loss_delta_of_generation(mutation_ops, crossover_ops)
        assert "SimpleMutation" in result
        assert result["SimpleMutation"] == [8.0 - 10.0]

    def test_compute_loss_delta_crossover(self):
        ap = AdaptiveProbabilities.__new__(AdaptiveProbabilities)
        ap.loss_index = {1: 10.0, 2: 12.0, 3: 9.0}
        mutation_ops = []
        crossover_ops = [{"parent1_genome": 1, "parent2_genome": 2, "child_id": 3, "operation": "AverageCrossover"}]
        result = ap.compute_loss_delta_of_generation(mutation_ops, crossover_ops)
        assert "AverageCrossover" in result
        min_parent = min(10.0, 12.0)
        assert result["AverageCrossover"] == [9.0 - min_parent]


class TestNuevolution:
    """Tests for Nuevolution evolution runner."""

    @patch("didgelab.evo.evolution.get_app")
    def test_evolve_returns_population(self, mock_get_app):
        mock_app = MagicMock()
        mock_app.publish = MagicMock()
        mock_app.subscribe = MagicMock()
        mock_app.get_service = MagicMock(return_value=None)
        mock_get_app.return_value = mock_app

        loss = TestLossFunction()
        father = DummyGenome(n_genes=6)
        evo = Nuevolution(
            loss=loss,
            father_genome=father,
            generation_size=2,
            num_generations=2,
            population_size=4,
        )
        population = evo.evolve()
        assert len(population) <= 4
        assert all(hasattr(p, "loss") and p.loss is not None for p in population)
        assert population == sorted(population, key=lambda x: x.loss["total"])

    @patch("didgelab.evo.evolution.get_app")
    def test_get_evolution_progress(self, mock_get_app):
        mock_app = MagicMock()
        mock_app.publish = MagicMock()
        mock_app.subscribe = MagicMock()
        mock_app.get_service = MagicMock(return_value=None)
        mock_get_app.return_value = mock_app

        loss = TestLossFunction()
        father = DummyGenome(n_genes=4)
        evo = Nuevolution(
            loss=loss,
            father_genome=father,
            generation_size=1,
            num_generations=5,
            population_size=2,
        )
        assert evo.get_evolution_progress() == 0.0
        evo.i_generation = 2
        assert evo.get_evolution_progress() == pytest.approx(0.6)
