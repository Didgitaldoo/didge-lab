"""
Pytest unit tests for didgelab.loss.loss (CompositeTairuaLoss and loss components).
"""

import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

pytest.importorskip("configargparse")

# Allow sim to load without Cython extension
_mock_cadsd = MagicMock()

def _mock_create_segments_from_geo(geo):
    from didgelab.sim.tlm_python import Segment
    return Segment.create_segments_from_geo(geo)


def _mock_cadsd_Ze(segments, freq):
    return 1e6


_mock_cadsd.create_segments_from_geo = _mock_create_segments_from_geo
_mock_cadsd.cadsd_Ze = _mock_cadsd_Ze
sys.modules["didgelab.sim.tlm_cython_lib._cadsd"] = _mock_cadsd

from didgelab.loss.loss import (
    CompositeTairuaLoss,
    FrequencyTuningLoss,
    ScaleTuningLoss,
    PeakQuantityLoss,
    PeakAmplitudeLoss,
    QFactorLoss,
    ModalDensityLoss,
    IntegerHarmonicLoss,
    NearIntegerLoss,
    StretchedOddLoss,
    HighInharmonicLoss,
    HarmonicSplittingLoss,
)
from didgelab.shapes.KigaliShape import KigaliShape


EXPECTED_COMPONENT_KEYS = [
    "freq",
    "scale",
    "peaks_qty",
    "peaks_amp",
    "q_factor",
    "modal_density",
    "integer_harmonic",
    "near_integer",
    "stretched_odd",
    "high_inharmonic",
    "harmonic_splitting",
]


def _build_full_composite_loss():
    """Build a CompositeTairuaLoss with all loss components."""
    target_freqs_hz = np.array([73.4, 146.8])
    target_freqs_log = np.log2(target_freqs_hz)

    loss = CompositeTairuaLoss(max_error=5.0)

    target_impedances = np.full(2, -1.0)  # frequency-only
    loss.add_component("freq", FrequencyTuningLoss(target_freqs_log, target_impedances, weights=[1.0, 1.0]))
    loss.add_component("scale", ScaleTuningLoss(base_note=60, intervals=[0, 2, 4, 5, 7, 9, 11], weight=5.0))
    loss.add_component("peaks_qty", PeakQuantityLoss(target_count=4, weight=2.0))
    loss.add_component("peaks_amp", PeakAmplitudeLoss(target_min_amplitude=0.25, weight=10.0))
    loss.add_component("q_factor", QFactorLoss(target_q=15.0, weight=1.0))
    loss.add_component("modal_density", ModalDensityLoss(cluster_range_cents=50.0, weight=1.0))
    loss.add_component("integer_harmonic", IntegerHarmonicLoss(weight=1.0))
    loss.add_component("near_integer", NearIntegerLoss(stretch_factor=1.002, weight=1.0))
    loss.add_component("stretched_odd", StretchedOddLoss(weight=1.0))
    loss.add_component("high_inharmonic", HighInharmonicLoss(weight=1.0))
    loss.add_component("harmonic_splitting", HarmonicSplittingLoss(harmonic_index=1, split_width_hz=5.0, weight=1.0))

    return loss


def _make_impedance_with_peaks(peak_freqs_hz):
    """Create impedance array with clear peaks at given frequencies (for mocking)."""
    from didgelab.acoustical_simulation import get_log_simulation_frequencies

    freq_grid = get_log_simulation_frequencies(1, 1000, 5.0)
    impedances = np.ones(len(freq_grid)) * 1e5
    for f_hz in peak_freqs_hz:
        idx = np.argmin(np.abs(freq_grid - f_hz))
        impedances[idx] = 1e7
        if idx > 0:
            impedances[idx - 1] = 0.5e7
        if idx < len(impedances) - 1:
            impedances[idx + 1] = 0.5e7
    return impedances


class TestCompositeTairuaLossFullExample:
    """Tests for CompositeTairuaLoss with all loss components."""

    def test_loss_contains_keys_for_all_components_and_total(self):
        """Assert that loss(shape) returns a dict with keys for all components and 'total'."""
        loss = _build_full_composite_loss()
        shape = KigaliShape(n_segments=20)
        shape.loss = None  # Ensure we compute, not use cache

        impedances = _make_impedance_with_peaks([73.4, 146.8])

        with patch("didgelab.loss.loss.acoustical_simulation", return_value=impedances):
            result = loss.loss(shape)

        assert "total" in result, "Result must contain 'total' key"
        for key in EXPECTED_COMPONENT_KEYS:
            assert key in result, f"Result must contain component key '{key}'"

        assert len(result) == len(EXPECTED_COMPONENT_KEYS) + 1
        assert isinstance(result["total"], (int, float))
        assert result["total"] >= 0
