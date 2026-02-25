"""
Pytest unit tests for didgelab.shapes.KigaliShape.
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
from didgelab.shapes.KigaliShape import KigaliShape


class TestKigaliShapeInit:
    """Tests for KigaliShape.__init__."""

    def test_genome_length_no_bubbles(self):
        shape = KigaliShape(n_segments=4, n_bubbles=0)
        expected = 3 + 2 * (4 - 1) + 0
        assert len(shape.genome) == expected

    def test_genome_length_with_bubbles(self):
        shape = KigaliShape(n_segments=4, n_bubbles=2)
        expected = 3 + 2 * (4 - 1) + 2 * 3
        assert len(shape.genome) == expected

    def test_attributes_stored(self):
        shape = KigaliShape(
            n_segments=5,
            d0=30,
            d_bell_min=40,
            d_bell_max=70,
            max_length=1800,
            min_length=1400,
        )
        assert shape.n_segments == 5
        assert shape.d0 == 30
        assert shape.d_bell_min == 40
        assert shape.d_bell_max == 70
        assert shape.max_length == 1800
        assert shape.min_length == 1400


class TestKigaliShapeGetProperties:
    """Tests for KigaliShape.get_properties."""

    def test_returns_six_tuple(self):
        shape = KigaliShape(n_segments=4, n_bubbles=0)
        out = shape.get_properties()
        assert len(out) == 6
        length, bell_size, power, x_genome, y_genome, bubbles = out
        assert length >= shape.min_length and length <= shape.max_length
        assert bell_size >= shape.d_bell_min and bell_size <= shape.d_bell_max
        assert power >= 0
        assert len(bubbles) == 0

    def test_bubbles_decoded_when_n_bubbles_nonzero(self):
        shape = KigaliShape(n_segments=4, n_bubbles=1)
        shape.genome = np.clip(shape.genome, 0.01, 0.99)
        _, _, _, _, _, bubbles = shape.get_properties()
        assert len(bubbles) == 1
        pos, width, height = bubbles[0]
        assert width >= 0
        assert height >= 0


class TestKigaliShapeGenome2Geo:
    """Tests for KigaliShape.genome2geo."""

    def test_returns_geo(self):
        shape = KigaliShape(n_segments=4, n_bubbles=0)
        geo = shape.genome2geo()
        assert isinstance(geo, Geo)
        assert len(geo.geo) >= 2

    def test_geo_x_strictly_increasing(self):
        shape = KigaliShape(n_segments=4, n_bubbles=0)
        geo = shape.genome2geo()
        x = [p[0] for p in geo.geo]
        assert x == sorted(x)

    def test_geo_diameters_in_reasonable_range(self):
        shape = KigaliShape(n_segments=4, n_bubbles=0)
        geo = shape.genome2geo()
        for _, d in geo.geo:
            assert d >= shape.d0 * 0.9
            assert d <= shape.d_bell_max * 1.3


class TestKigaliShapeFixDidge:
    """Tests for KigaliShape.fix_didge."""

    def test_clamps_below_min(self):
        shape = KigaliShape(n_segments=4, n_bubbles=0)
        x = np.array([0.0, 500.0, 1000.0])
        y = np.array([20.0, 25.0, 60.0])
        x_out, y_out = shape.fix_didge(x, y, d0=32, bellsize=60)
        assert np.all(y_out >= 32 * 0.9)

    def test_clamps_above_max(self):
        shape = KigaliShape(n_segments=4, n_bubbles=0)
        x = np.array([0, 500, 1000])
        y = np.array([32, 80, 100])
        x_out, y_out = shape.fix_didge(x, y, d0=32, bellsize=60)
        assert np.all(y_out <= 60 * 1.3)

    def test_x_unchanged(self):
        shape = KigaliShape(n_segments=4, n_bubbles=0)
        x = np.array([0, 500, 1000], dtype=float)
        y = np.array([32, 50, 60], dtype=float)
        x_out, _ = shape.fix_didge(x, y, d0=32, bellsize=60)
        np.testing.assert_array_equal(x_out, x)


class TestKigaliShapeMakeBubble:
    """Tests for KigaliShape.make_bubble."""

    def test_increases_number_of_points(self):
        shape = KigaliShape(n_segments=4, n_bubbles=0)
        x = np.array([0, 400, 800, 1200], dtype=float)
        y = np.array([32, 40, 50, 60], dtype=float)
        x_new, y_new = shape.make_bubble(x, y, pos=600, width=200, height=5)
        assert len(x_new) > len(x)
        assert len(y_new) == len(x_new)

    def test_output_ordered_by_x(self):
        shape = KigaliShape(n_segments=4, n_bubbles=0)
        x = np.linspace(0, 1000, 5)
        y = np.linspace(32, 60, 5)
        x_new, y_new = shape.make_bubble(x, y, pos=500, width=100, height=3)
        assert np.all(np.diff(x_new) >= 0)
