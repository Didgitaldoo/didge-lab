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

class TestKigaliShapeFixDidge:
    """Tests for KigaliShape.fix_didge."""


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


class TestKigaliShapeEnableCylinderCone:
    """Tests for KigaliShape enable_cylinder_cone setting."""

    def test_genome_length_increases_by_two_when_enabled(self):
        base = KigaliShape(n_segments=4, n_bubbles=0, enable_cylinder_cone=False)
        with_cyl = KigaliShape(n_segments=4, n_bubbles=0, enable_cylinder_cone=True)
        assert len(with_cyl.genome) == len(base.genome) + 2

    def test_cylinder_cone_attributes_stored(self):
        shape = KigaliShape(
            enable_cylinder_cone=True,
            cylinder_cone_z_pos=[0.2, 0.8],
            cylinder_cone_transition_diameter=[30, 40],
        )
        assert shape.enable_cylinder_cone is True
        assert shape.cylinder_cone_z_pos == [0.2, 0.8]
        assert shape.cylinder_cone_transition_diameter == [30, 40]
        assert hasattr(shape, "cyliner_cone_genome_offset")

    def test_genome2geo_returns_valid_geo_when_enabled(self):
        shape = KigaliShape(
            n_segments=8,
            n_bubbles=0,
            enable_cylinder_cone=True,
        )
        geo = shape.genome2geo()
        assert isinstance(geo, Geo)
        assert len(geo.geo) >= 2
        x = [p[0] for p in geo.geo]
        assert x == sorted(x)
        for _, d in geo.geo:
            assert d >= shape.d0 * 0.5
            assert d <= shape.d_bell_max * 1.3

    def test_create_cylinder_cone_y_cylindrical_then_conical(self):
        """Cylinder-cone profile: cylindrical section to transition, then conical to bell."""
        shape = KigaliShape(
            n_segments=20,
            n_bubbles=0,
            enable_cylinder_cone=True,
            cylinder_cone_z_pos=[0.3, 0.3],
            cylinder_cone_transition_diameter=[34, 34],
        )
        shape.genome[shape.cyliner_cone_genome_offset] = 0.0
        shape.genome[shape.cyliner_cone_genome_offset + 1] = 0.0
        bell_size = 60.0
        y = shape.create_cylinder_cone_y(bell_size)
        z = np.linspace(0, 1, shape.n_segments + 1)
        assert len(y) == len(z)
        assert y[0] == shape.d0
        transition_z = 0.3
        transition_d = 34.0
        idx_transition = np.argmin(np.abs(z - transition_z))
        assert np.isclose(y[idx_transition], transition_d, atol=2)
        assert np.isclose(y[-1], bell_size, atol=1)

    def test_cylinder_cone_vs_power_law_different_output(self):
        """Enable cylinder-cone produces a different geometry than power-law."""
        shape_cyl = KigaliShape(n_segments=6, n_bubbles=0, enable_cylinder_cone=True)
        shape_power = KigaliShape(n_segments=6, n_bubbles=0, enable_cylinder_cone=False)
        np.random.seed(42)
        shape_cyl.genome = np.random.uniform(0.1, 0.9, len(shape_cyl.genome))
        shape_power.genome = shape_cyl.genome[: len(shape_power.genome)].copy()
        geo_cyl = shape_cyl.genome2geo()
        geo_power = shape_power.genome2geo()
        y_cyl = [p[1] for p in geo_cyl.geo]
        y_power = [p[1] for p in geo_power.geo]
        assert y_cyl != y_power


class TestKigaliShapeEnableTaper:
    """Tests for KigaliShape enable_taper setting."""

    def test_taper_attributes_stored(self):
        shape = KigaliShape(
            enable_taper=True,
            taper_start=20,
            taper_end=50,
            taper_diameter=20,
            taper_n_segments=8,
        )
        assert shape.enable_taper is True
        assert shape.taper_start == 20
        assert shape.taper_end == 50
        assert shape.taper_diameter == 20
        assert shape.taper_n_segments == 8

    def test_genome2geo_returns_valid_geo_when_taper_enabled(self):
        shape = KigaliShape(
            n_segments=8,
            n_bubbles=0,
            enable_taper=True,
            taper_start=30,
            taper_end=80,
        )
        geo = shape.genome2geo()
        assert isinstance(geo, Geo)
        assert len(geo.geo) >= 2
        x = [p[0] for p in geo.geo]
        assert x == sorted(x)
        for _, d in geo.geo:
            assert d >= 0
            assert d <= shape.d_bell_max * 1.3

    def test_apply_mouthpiece_taper_creates_bottleneck(self):
        """Taper creates restriction: diameter narrows between taper_start and taper_end."""
        shape = KigaliShape(
            n_segments=16,
            n_bubbles=0,
            enable_taper=True,
            d0=32,
            taper_start=50,
            taper_end=150,
            taper_diameter=22,
        )
        x = np.linspace(0, 1000, 20)
        y = np.linspace(32, 60, 20)
        x_new, y_new = shape.apply_mouthpiece_taper(x, y)
        assert len(x_new) > len(x)
        idx_start = np.argmin(np.abs(x_new - shape.taper_start))
        idx_mid = np.argmin(np.abs(x_new - (shape.taper_start + shape.taper_end) / 2))
        idx_end = np.argmin(np.abs(x_new - shape.taper_end))
        assert np.isclose(y_new[idx_start], shape.d0, atol=2)
        assert np.isclose(y_new[idx_mid], shape.taper_diameter, atol=3)

    def test_taper_vs_no_taper_different_output(self):
        """Enable taper produces a different geometry than no taper."""
        shape_taper = KigaliShape(n_segments=8, n_bubbles=0, enable_taper=True)
        shape_no_taper = KigaliShape(n_segments=8, n_bubbles=0, enable_taper=False)
        np.random.seed(43)
        g = np.random.uniform(0.2, 0.8, len(shape_taper.genome))
        shape_taper.genome = g.copy()
        shape_no_taper.genome = g.copy()
        geo_taper = shape_taper.genome2geo()
        geo_no_taper = shape_no_taper.genome2geo()
        y_taper = [p[1] for p in geo_taper.geo]
        y_no_taper = [p[1] for p in geo_no_taper.geo]
        assert y_taper != y_no_taper
