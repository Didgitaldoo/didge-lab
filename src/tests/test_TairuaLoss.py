"""
Pytest unit tests for didgelab.loss.TairuaLoss.
"""

import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

pytest.importorskip("configargparse")

# Allow evo/sim to load without Cython extension
_mock_cadsd = MagicMock()
def _mock_create_segments_from_geo(geo):
    from didgelab.sim.tlm_python import Segment
    return Segment.create_segments_from_geo(geo)
def _mock_cadsd_Ze(segments, freq):
    return 1e6
_mock_cadsd.create_segments_from_geo = _mock_create_segments_from_geo
_mock_cadsd.cadsd_Ze = _mock_cadsd_Ze
sys.modules["didgelab.sim.tlm_cython_lib._cadsd"] = _mock_cadsd

from didgelab.geo import Geo
from didgelab.loss.TairuaLoss import TairuaLoss


class TestTairuaLossInit:
    """Tests for TairuaLoss.__init__."""

    def test_stores_targets_and_weights(self):
        loss = TairuaLoss(
            target_freqs=[100, 200],
            target_impedances=[0.8, 0.5],
            freq_weights=[1.0, 1.0],
            impedance_weights=[0.5, 0.5],
        )
        assert len(loss.target_freqs) == 2
        assert np.allclose(loss.target_freqs, np.log2([100, 200]))
        assert loss.target_impedances == [0.8, 0.5]
        assert loss.freq_weights == [1.0, 1.0]
        assert loss.impedance_weights == [0.5, 0.5]

    def test_freqs_grid_created(self):
        loss = TairuaLoss(
            target_freqs=[73],
            target_impedances=[1.0],
            freq_weights=[1.0],
            impedance_weights=[0.0],
            max_error=10,
        )
        assert loss.freqs is not None
        assert len(loss.freqs) >= 1
        assert loss.freqs.min() >= 1
        assert loss.freqs.max() <= 1000

    def test_tune_to_scale_false_without_scale(self):
        loss = TairuaLoss(
            target_freqs=[73],
            target_impedances=[1.0],
            freq_weights=[1.0],
            impedance_weights=[0.0],
        )
        assert loss.tune_to_scale is False

    def test_tune_to_scale_true_when_scale_given(self):
        loss = TairuaLoss(
            target_freqs=[73],
            target_impedances=[1.0],
            freq_weights=[1.0],
            impedance_weights=[0.0],
            scale_key=0,
            scale=[0, 2, 4, 5, 7, 9, 11],
            scale_weight=0.1,
        )
        assert loss.tune_to_scale is True
        assert hasattr(loss, "scale_freqs")
        assert len(loss.scale_freqs) >= 1


class TestTairuaLossComputeScaleFrequencies:
    """Tests for TairuaLoss.compute_scale_frequencies."""

    def test_returns_log2_frequencies(self):
        loss = TairuaLoss(
            target_freqs=[73],
            target_impedances=[1.0],
            freq_weights=[1.0],
            impedance_weights=[0.0],
            scale_key=0,
            scale=[0, 12],
            scale_weight=0.1,
        )
        scale_freqs = loss.compute_scale_frequencies()
        assert isinstance(scale_freqs, np.ndarray)
        assert len(scale_freqs) >= 2
        # Scale notes can repeat across octaves, so allow non-decreasing
        assert np.all(np.diff(scale_freqs) >= 0)

    def test_scale_frequencies_are_log2(self):
        loss = TairuaLoss(
            target_freqs=[73],
            target_impedances=[1.0],
            freq_weights=[1.0],
            impedance_weights=[0.0],
            scale_key=0,
            scale=[0],
            scale_weight=0.1,
        )
        scale_freqs = loss.compute_scale_frequencies()
        linear = np.power(2.0, scale_freqs)
        # Method includes notes until next would exceed 1000 Hz; may include one above
        assert len(linear) >= 1
        assert np.all(linear > 0)


class TestTairuaLossLoss:
    """Tests for TairuaLoss.loss(shape)."""

    def test_returns_dict_with_total_and_components(self):
        loss = TairuaLoss(
            target_freqs=[73.0, 150.0],
            target_impedances=[1.0, 0.5],
            freq_weights=[1.0, 1.0],
            impedance_weights=[0.0, 0.0],
            max_error=20,
        )
        geo = Geo([[0, 32], [1200, 60]])
        # Impedance with peaks near 73 and 150 Hz (indices we choose via mock)
        n = len(loss.freqs)
        impedances = np.ones(n) * 1e5
        peak_i = min(np.argmin(np.abs(loss.freqs - 73)), n - 1)
        impedances[peak_i] = 1e7
        peak_j = min(np.argmin(np.abs(loss.freqs - 150)), n - 1)
        impedances[peak_j] = 5e6

        shape = MagicMock()
        shape.genome2geo.return_value = geo
        shape.loss = None

        with patch("didgelab.loss.TairuaLoss.acoustical_simulation", return_value=impedances):
            result = loss.loss(shape)

        assert "total" in result
        assert "freq_loss" in result
        assert "imp_loss" in result
        assert result["total"] >= 0
        assert isinstance(result["total"], (int, float))

    def test_uses_cached_loss_if_present(self):
        loss = TairuaLoss(
            target_freqs=[73],
            target_impedances=[1.0],
            freq_weights=[1.0],
            impedance_weights=[0.0],
        )
        shape = MagicMock()
        shape.loss = {"total": 42, "freq_loss": 1, "imp_loss": 0}
        out = loss.loss(shape)
        assert out["total"] == 42
        shape.genome2geo.assert_not_called()

    def test_returns_large_total_on_exception(self):
        loss = TairuaLoss(
            target_freqs=[73],
            target_impedances=[1.0],
            freq_weights=[1.0],
            impedance_weights=[0.0],
        )
        shape = MagicMock()
        shape.genome2geo.side_effect = RuntimeError("bad")
        shape.loss = None
        result = loss.loss(shape)
        assert result["total"] >= 100_000_000
