"""
Pytest unit tests for didgelab.geo.
"""

import tempfile
import json
import os
import pytest
import numpy as np

from didgelab.geo import Geo


class TestGeoInit:
    """Tests for Geo.__init__."""

    def test_empty_geo(self):
        g = Geo()
        assert g.geo == []

    def test_from_list(self):
        geo = [[0, 32], [600, 45], [1200, 60]]
        g = Geo(geo=geo)
        assert len(g.geo) == 3
        assert g.geo[0] == [0, 32]
        assert g.geo[-1] == [1200, 60]

    def test_removes_zero_length_segments(self):
        geo = [[0, 32], [600, 45], [600, 50], [1200, 60]]  # duplicate x at 600
        g = Geo(geo=geo)
        assert len(g.geo) == 3
        assert g.geo[1][0] == 600


class TestMakeCone:
    """Tests for Geo.make_cone."""

    def test_cone_segments(self):
        g = Geo.make_cone(length=1200, d1=32, d2=60, n_segments=5)
        assert len(g.geo) == 5
        assert g.geo[0] == [0, 32]
        assert g.geo[-1][0] == 1200
        assert g.geo[-1][1] == pytest.approx(60, abs=1e-9)

    def test_cone_monotonic_diameter(self):
        g = Geo.make_cone(length=1000, d1=30, d2=80, n_segments=10)
        diameters = [s[1] for s in g.geo]
        assert diameters == sorted(diameters)


class TestGeoFileIo:
    """Tests for read_geo and write_geo."""

    def test_write_geo_produces_expected_format(self):
        geo = [[0, 32], [600, 45], [1200, 60]]
        g = Geo(geo=geo)
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False, mode="w") as f:
            path = f.name
        try:
            g.write_geo(path)
            with open(path) as fp:
                lines = fp.readlines()
            assert len(lines) == 3
            assert "32" in lines[0] and "60" in lines[-1]
        finally:
            os.unlink(path)

    def test_read_json_geo(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
            json.dump([[0, 32], [1200, 60]], f)
            path = f.name
        try:
            g = Geo(infile=path)
            assert len(g.geo) == 2
            assert g.geo[0] == [0, 32]
            assert g.geo[1] == [1200, 60]
        finally:
            os.unlink(path)


class TestGeoTransformations:
    """Tests for stretch, scale, copy."""

    def test_stretch_scales_x_only(self):
        g = Geo(geo=[[0, 32], [1200, 60]])
        g.stretch(2.0)
        assert g.geo[0][0] == 0
        assert g.geo[1][0] == 2400
        assert g.geo[0][1] == 32
        assert g.geo[1][1] == 60

    def test_scale_scales_both(self):
        g = Geo(geo=[[0, 32], [1200, 60]])
        g.scale(0.001)  # mm to m
        assert g.geo[0][0] == 0
        assert g.geo[1][0] == pytest.approx(1.2)
        assert g.geo[0][1] == pytest.approx(0.032)
        assert g.geo[1][1] == pytest.approx(0.06)

    def test_copy_is_independent(self):
        g = Geo(geo=[[0, 32], [1200, 60]])
        g2 = g.copy()
        g2.stretch(2.0)
        assert g.geo[1][0] == 1200
        assert g2.geo[1][0] == 2400


class TestMakeBubble:
    """Tests for make_bubble."""

    def test_bubble_adds_segments(self):
        g = Geo(geo=[[0, 32], [600, 40], [1200, 60]])
        g.make_bubble(pos=400, width=100, height=80)
        assert len(g.geo) >= 3
        # Should have segments around pos with height 80
        heights = [s[1] for s in g.geo]
        assert 80 in heights or max(heights) >= 80


class TestMoveSegmentsX:
    """Tests for move_segments_x."""

    def test_shifts_segment_x(self):
        g = Geo(geo=[[0, 32], [400, 40], [1200, 60]])
        g.move_segments_x(1, 1, 50)
        assert g.geo[1][0] == 450
        assert g.geo[0][0] == 0
        assert g.geo[2][0] == 1200


class TestGeoProperties:
    """Tests for length, bellsize, segments_to_str."""

    def test_length(self):
        g = Geo(geo=[[0, 32], [1200, 60]])
        assert g.length() == 1200

    def test_bellsize(self):
        g = Geo(geo=[[0, 32], [1200, 60]])
        assert g.bellsize() == 60

    def test_segments_to_str(self):
        g = Geo(geo=[[0, 32], [1200, 60]])
        s = g.segments_to_str()
        assert "32" in s and "60" in s


class TestSortSegments:
    """Tests for sort_segments."""

    def test_sort_by_x(self):
        g = Geo(geo=[[1200, 60], [0, 32], [600, 45]])
        g.sort_segments()
        assert g.geo[0][0] == 0
        assert g.geo[1][0] == 600
        assert g.geo[2][0] == 1200


class TestDiameterAtX:
    """Tests for diameter_at_x (static and instance)."""

    def test_at_start(self):
        g = Geo(geo=[[0, 32], [1200, 60]])
        assert Geo.diameter_at_x(g, 0) == 32

    def test_at_end(self):
        g = Geo(geo=[[0, 32], [1200, 60]])
        assert Geo.diameter_at_x(g, 1200) == 60

    def test_interpolation_midway(self):
        g = Geo(geo=[[0, 32], [1200, 60]])
        d = Geo.diameter_at_x(g, 600)
        assert 32 < d < 60

    def test_instance_method_delegates_to_static(self):
        g = Geo(geo=[[0, 32], [1200, 60]])
        # Instance method delegates to static: Geo.diameter_at_x(geo, x)
        assert Geo.diameter_at_x(g, 0) == 32


class TestComputeVolume:
    """Tests for compute_volume."""

    def test_cylinder_volume(self):
        # Cylinder: 1200 mm long, 60 mm diameter
        # Geo uses trapezoidal rule: v += l*d0 + l*(d1-d0) = l*d1 for constant d
        g = Geo(geo=[[0, 60], [1200, 60]])
        v = g.compute_volume()
        assert v > 0
        assert v == 1200 * 60  # length * diameter for constant bore

    def test_cone_positive_volume(self):
        g = Geo.make_cone(length=1200, d1=32, d2=60, n_segments=5)
        v = g.compute_volume()
        assert v > 0


class TestStaticMethods:
    """Tests for scale_length, get_max_d, scale_diameter."""

    def test_scale_length(self):
        g = Geo(geo=[[0, 32], [1200, 60]])
        Geo.scale_length(g, 2400)
        assert g.length() == 2400

    def test_get_max_d(self):
        g = Geo(geo=[[0, 32], [600, 80], [1200, 60]])
        assert Geo.get_max_d(g) == 80

    def test_scale_diameter(self):
        g = Geo(geo=[[0, 32], [1200, 60]])
        Geo.scale_diameter(g, 100)
        assert Geo.get_max_d(g) == pytest.approx(100, abs=1e-9)


class TestGeoToJson:
    """Tests for geo_to_json and json_to_geo."""

    def test_roundtrip(self):
        geo = [[0, 32], [1200, 60]]
        g = Geo(geo=geo)
        j = Geo.geo_to_json(g)
        assert j == geo
        g2 = Geo.json_to_geo(j)
        assert g2.geo == geo


class TestFixZeroLengthSegments:
    """Tests for fix_zero_length_segments."""

    def test_removes_duplicate_x(self):
        g = Geo(geo=[[0, 32], [600, 45], [600, 50], [1200, 60]])
        fixed = Geo.fix_zero_length_segments(g)
        xs = [s[0] for s in fixed.geo]
        assert xs == sorted(set(xs))
        assert len(fixed.geo) == 3
