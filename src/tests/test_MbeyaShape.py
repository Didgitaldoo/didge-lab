"""
Pytest unit tests for didgelab.shapes.MbeyaShape (MbeyaGenome).
"""

import sys
from unittest.mock import MagicMock

import numpy as np
import pytest

pytest.importorskip("configargparse")

_mock_cadsd = MagicMock()
def _mock_create_segments_from_geo(geo):
    from didgelab.sim.tlm_python import Segment
    return Segment.create_segments_from_geo(geo)
_mock_cadsd.create_segments_from_geo = _mock_create_segments_from_geo
_mock_cadsd.cadsd_Ze = lambda s, f: 1e6
sys.modules["didgelab.sim.tlm_cython_lib._cadsd"] = _mock_cadsd

from didgelab.geo import Geo
from didgelab.shapes.MbeyaShape import MbeyaGenome


class TestMbeyaGenomeInit:
    """Tests for MbeyaGenome.__init__."""

    def test_genome_length_matches_named_params(self):
        gen = MbeyaGenome(n_bubbles=1)
        assert len(gen.genome) == len(gen.named_params)

    def test_n_bubbles_stored(self):
        gen = MbeyaGenome(n_bubbles=5)
        assert gen.n_bubbles == 5


class TestMbeyaGenomeAddParamGetValue:
    """Tests for add_param and get_value."""

    def test_get_value_in_range(self):
        gen = MbeyaGenome(n_bubbles=0)
        gen.genome = np.zeros(len(gen.genome))
        v = gen.get_value("l_gerade")
        assert v == 500
        gen.genome[gen.named_params["l_gerade"]["index"]] = 1.0
        v = gen.get_value("l_gerade")
        assert v == 1500

    def test_get_value_interpolates(self):
        gen = MbeyaGenome(n_bubbles=0)
        idx = gen.named_params["l_gerade"]["index"]
        gen.genome[idx] = 0.5
        v = gen.get_value("l_gerade")
        assert 500 < v < 1500
        assert abs(v - 1000) < 10


class TestMbeyaGenomeGetIndex:
    """Tests for get_index."""

    def test_returns_first_index_where_x_exceeds(self):
        gen = MbeyaGenome(n_bubbles=0)
        shape = [[0, 32], [500, 35], [1000, 50], [1500, 60]]
        assert gen.get_index(shape, 100) == 1
        assert gen.get_index(shape, 0) == 1
        assert gen.get_index(shape, 600) == 2
        assert gen.get_index(shape, 1400) == 3

    def test_returns_len_minus_one_if_x_above_all(self):
        gen = MbeyaGenome(n_bubbles=0)
        shape = [[0, 32], [1000, 60]]
        assert gen.get_index(shape, 2000) == 1


class TestMbeyaGenomeMakeBubble:
    """Tests for make_bubble."""

    def test_returns_list_of_points(self):
        gen = MbeyaGenome(n_bubbles=0)
        shape = [[0, 32], [500, 40], [1000, 50]]
        out = gen.make_bubble(shape, pos=500, width=200, height=0.2)
        assert isinstance(out, list)
        assert all(len(p) == 2 for p in out)
        assert len(out) > len(shape)

    def test_x_values_increasing(self):
        gen = MbeyaGenome(n_bubbles=0)
        shape = [[0, 32], [400, 38], [800, 48], [1200, 55]]
        out = gen.make_bubble(shape, pos=600, width=150, height=0.1)
        x = [p[0] for p in out]
        assert x == sorted(x)


class TestMbeyaGenomeGenome2Geo:
    """Tests for MbeyaGenome.genome2geo."""

    def test_returns_geo(self):
        gen = MbeyaGenome(n_bubbles=0)
        geo = gen.genome2geo()
        assert isinstance(geo, Geo)
        assert len(geo.geo) >= 2

    def test_geo_starts_at_mouth(self):
        gen = MbeyaGenome(n_bubbles=0)
        geo = gen.genome2geo()
        assert geo.geo[0][0] == 0
        assert geo.geo[0][1] == gen.d1

    def test_geo_x_increasing(self):
        gen = MbeyaGenome(n_bubbles=0)
        gen.genome = np.clip(gen.genome, 0.01, 0.99)
        geo = gen.genome2geo()
        x = [p[0] for p in geo.geo]
        assert x == sorted(x)


