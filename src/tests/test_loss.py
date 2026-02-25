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


def _make_dummy_spectrum(n_pts=100, peak_indices=None):
    """Create a minimal frequency grid and impedance array with peaks for component tests."""
    from didgelab.acoustical_simulation import get_log_simulation_frequencies

    freq_grid = get_log_simulation_frequencies(1, 1000, 5.0)
    impedances = np.ones(len(freq_grid)) * 1e5
    if peak_indices is None:
        peak_indices = [10, 30, 50, 70]
    for i in peak_indices:
        i = min(max(0, i), len(impedances) - 1)
        impedances[i] = 1e7
    return freq_grid, impedances


class TestFrequencyTuningLoss:
    def test_calculate_returns_non_negative(self):
        target_log = np.log2(np.array([80.0, 160.0]))
        loss = FrequencyTuningLoss(target_log, np.array([-1.0, -1.0]), [1.0, 1.0])
        peak_log = np.log2(np.array([80.0, 160.0]))
        peak_imp = np.array([0.8, 0.8])
        freq_grid, impedances = _make_dummy_spectrum(peak_indices=[10, 30])
        peak_idx = np.array([10, 30])
        val = loss.calculate(peak_log, peak_imp, freq_grid, impedances, peak_idx)
        assert val >= 0
        assert isinstance(val, (int, float))

    def test_calculate_low_loss_when_peaks_match_targets(self):
        target_log = np.log2(np.array([80.0, 160.0]))
        loss = FrequencyTuningLoss(target_log, np.array([-1.0, -1.0]), [1.0, 1.0])
        peak_log = np.log2(np.array([80.0, 160.0]))
        peak_imp = np.array([0.8, 0.8])
        freq_grid, impedances = _make_dummy_spectrum(peak_indices=[10, 30])
        peak_idx = np.array([10, 30])
        val = loss.calculate(peak_log, peak_imp, freq_grid, impedances, peak_idx)
        assert val < 0.1

    def test_calculate_higher_loss_when_peaks_mismatch(self):
        target_log = np.log2(np.array([80.0, 160.0]))
        loss = FrequencyTuningLoss(target_log, np.array([-1.0, -1.0]), [1.0, 1.0])
        peak_log = np.log2(np.array([90.0, 180.0]))
        peak_imp = np.array([0.8, 0.8])
        freq_grid, impedances = _make_dummy_spectrum(peak_indices=[10, 30])
        peak_idx = np.array([10, 30])
        val = loss.calculate(peak_log, peak_imp, freq_grid, impedances, peak_idx)
        assert val > 0.1

    def test_get_formula_returns_tuple(self):
        loss = FrequencyTuningLoss(np.array([0.0]), np.array([-1.0]), [1.0])
        formula, symbols = loss.get_formula()
        assert isinstance(formula, str)
        assert isinstance(symbols, list)
        assert len(formula) > 0
        assert len(symbols) > 0


class TestScaleTuningLoss:
    def test_calculate_returns_non_negative(self):
        loss = ScaleTuningLoss(60, [0, 2, 4, 5, 7, 9, 11], 1.0)
        peak_log = np.log2(np.array([80.0, 160.0, 240.0]))
        peak_imp = np.array([0.8, 0.8, 0.8])
        freq_grid, impedances = _make_dummy_spectrum()
        peak_idx = np.array([10, 30, 50])
        val = loss.calculate(peak_log, peak_imp, freq_grid, impedances, peak_idx)
        assert val >= 0

    def test_get_formula_returns_tuple(self):
        loss = ScaleTuningLoss(60, [0, 2, 4], 1.0)
        formula, symbols = loss.get_formula()
        assert isinstance(formula, str)
        assert isinstance(symbols, list)


class TestPeakQuantityLoss:
    def test_calculate_zero_when_enough_peaks(self):
        loss = PeakQuantityLoss(target_count=3, weight=2.0)
        peak_log = np.log2(np.array([80.0, 120.0, 160.0]))
        peak_imp = np.ones(3) * 0.8
        freq_grid, impedances = _make_dummy_spectrum()
        peak_idx = np.array([10, 25, 40])
        val = loss.calculate(peak_log, peak_imp, freq_grid, impedances, peak_idx)
        assert val == 0

    def test_calculate_penalty_when_few_peaks(self):
        loss = PeakQuantityLoss(target_count=5, weight=2.0)
        peak_log = np.log2(np.array([80.0, 160.0]))
        peak_imp = np.ones(2) * 0.8
        freq_grid, impedances = _make_dummy_spectrum()
        peak_idx = np.array([10, 30])
        val = loss.calculate(peak_log, peak_imp, freq_grid, impedances, peak_idx)
        assert val == 3 * 2.0

    def test_get_formula_returns_tuple(self):
        loss = PeakQuantityLoss(5, 1.0)
        formula, symbols = loss.get_formula()
        assert isinstance(formula, str)
        assert isinstance(symbols, list)


class TestPeakAmplitudeLoss:
    def test_calculate_zero_when_amps_above_target(self):
        loss = PeakAmplitudeLoss(target_min_amplitude=0.2, weight=5.0)
        peak_log = np.log2(np.array([80.0, 160.0]))
        peak_imp = np.array([0.8, 0.9])
        freq_grid, impedances = _make_dummy_spectrum()
        peak_idx = np.array([10, 30])
        val = loss.calculate(peak_log, peak_imp, freq_grid, impedances, peak_idx)
        assert val == 0

    def test_calculate_penalty_when_amps_below_target(self):
        loss = PeakAmplitudeLoss(target_min_amplitude=0.8, weight=5.0)
        peak_log = np.log2(np.array([80.0, 160.0]))
        peak_imp = np.array([0.2, 0.3])
        freq_grid, impedances = _make_dummy_spectrum()
        peak_idx = np.array([10, 30])
        val = loss.calculate(peak_log, peak_imp, freq_grid, impedances, peak_idx)
        assert val > 0

    def test_get_formula_returns_tuple(self):
        loss = PeakAmplitudeLoss(0.25, 1.0)
        formula, symbols = loss.get_formula()
        assert isinstance(formula, str)
        assert isinstance(symbols, list)


class TestQFactorLoss:
    def test_calculate_returns_non_negative(self):
        loss = QFactorLoss(target_q=15.0, weight=1.0)
        peak_log = np.log2(np.array([80.0, 120.0, 160.0]))
        peak_imp = np.array([0.5, 0.8, 0.5])
        freq_grid, impedances = _make_dummy_spectrum()
        peak_idx = np.array([10, 25, 40])
        val = loss.calculate(peak_log, peak_imp, freq_grid, impedances, peak_idx)
        assert val >= 0
        assert isinstance(val, (int, float))

    def test_get_formula_returns_tuple(self):
        loss = QFactorLoss(15.0, 1.0)
        formula, symbols = loss.get_formula()
        assert isinstance(formula, str)
        assert isinstance(symbols, list)


class TestModalDensityLoss:
    def test_calculate_max_loss_when_single_peak(self):
        loss = ModalDensityLoss(cluster_range_cents=50.0, weight=1.0)
        peak_log = np.log2(np.array([80.0]))
        peak_imp = np.array([0.8])
        freq_grid, impedances = _make_dummy_spectrum()
        peak_idx = np.array([10])
        val = loss.calculate(peak_log, peak_imp, freq_grid, impedances, peak_idx)
        assert val == 1.0

    def test_calculate_returns_non_negative_with_multiple_peaks(self):
        loss = ModalDensityLoss(cluster_range_cents=50.0, weight=1.0)
        peak_log = np.log2(np.array([80.0, 85.0, 160.0]))
        peak_imp = np.ones(3) * 0.8
        freq_grid, impedances = _make_dummy_spectrum()
        peak_idx = np.array([10, 12, 30])
        val = loss.calculate(peak_log, peak_imp, freq_grid, impedances, peak_idx)
        assert val >= 0

    def test_get_formula_returns_tuple(self):
        loss = ModalDensityLoss(30.0, 1.0)
        formula, symbols = loss.get_formula()
        assert isinstance(formula, str)
        assert isinstance(symbols, list)


class TestIntegerHarmonicLoss:
    def test_calculate_zero_for_perfect_harmonics(self):
        loss = IntegerHarmonicLoss(weight=1.0)
        f0_hz = 100.0
        peak_log = np.log2(np.array([f0_hz, 2 * f0_hz, 3 * f0_hz]))
        peak_imp = np.ones(3) * 0.8
        freq_grid, impedances = _make_dummy_spectrum()
        peak_idx = np.array([10, 20, 30])
        val = loss.calculate(peak_log, peak_imp, freq_grid, impedances, peak_idx)
        assert val < 0.01

    def test_calculate_positive_for_inharmonic(self):
        loss = IntegerHarmonicLoss(weight=1.0)
        peak_log = np.log2(np.array([100.0, 250.0, 400.0]))
        peak_imp = np.ones(3) * 0.8
        freq_grid, impedances = _make_dummy_spectrum()
        peak_idx = np.array([10, 25, 40])
        val = loss.calculate(peak_log, peak_imp, freq_grid, impedances, peak_idx)
        assert val > 0

    def test_get_formula_returns_tuple(self):
        loss = IntegerHarmonicLoss(1.0)
        formula, symbols = loss.get_formula()
        assert isinstance(formula, str)
        assert isinstance(symbols, list)


class TestNearIntegerLoss:
    def test_calculate_returns_non_negative(self):
        loss = NearIntegerLoss(stretch_factor=1.002, weight=1.0)
        f0_hz = 100.0
        peak_log = np.log2(np.array([f0_hz, 2 * f0_hz * 1.002, 3 * f0_hz * (1.002 ** 2)]))
        peak_imp = np.ones(3) * 0.8
        freq_grid, impedances = _make_dummy_spectrum()
        peak_idx = np.array([10, 20, 30])
        val = loss.calculate(peak_log, peak_imp, freq_grid, impedances, peak_idx)
        assert val >= 0

    def test_get_formula_returns_tuple(self):
        loss = NearIntegerLoss(1.002, 1.0)
        formula, symbols = loss.get_formula()
        assert isinstance(formula, str)
        assert isinstance(symbols, list)


class TestStretchedOddLoss:
    def test_calculate_returns_non_negative(self):
        loss = StretchedOddLoss(weight=1.0)
        f0_hz = 100.0
        peak_log = np.log2(np.array([f0_hz, 3.1 * f0_hz, 5.2 * f0_hz]))
        peak_imp = np.ones(3) * 0.8
        freq_grid, impedances = _make_dummy_spectrum()
        peak_idx = np.array([10, 30, 50])
        val = loss.calculate(peak_log, peak_imp, freq_grid, impedances, peak_idx)
        assert val >= 0

    def test_get_formula_returns_tuple(self):
        loss = StretchedOddLoss(1.0)
        formula, symbols = loss.get_formula()
        assert isinstance(formula, str)
        assert isinstance(symbols, list)


class TestHighInharmonicLoss:
    def test_calculate_returns_non_negative_for_harmonic_peaks(self):
        loss = HighInharmonicLoss(weight=1.0)
        f0_hz = 100.0
        peak_log = np.log2(np.array([f0_hz, 2 * f0_hz, 3 * f0_hz]))
        peak_imp = np.ones(3) * 0.8
        freq_grid, impedances = _make_dummy_spectrum()
        peak_idx = np.array([10, 20, 30])
        val = loss.calculate(peak_log, peak_imp, freq_grid, impedances, peak_idx)
        assert val >= 0

    def test_get_formula_returns_tuple(self):
        loss = HighInharmonicLoss(1.0)
        formula, symbols = loss.get_formula()
        assert isinstance(formula, str)
        assert isinstance(symbols, list)


class TestHarmonicSplittingLoss:
    def test_calculate_zero_when_split_exists(self):
        loss = HarmonicSplittingLoss(harmonic_index=1, split_width_hz=20.0, weight=1.0)
        f_hz = 100.0
        peak_log = np.log2(np.array([f_hz, f_hz + 5, f_hz + 10, 200.0]))
        peak_imp = np.ones(4) * 0.8
        freq_grid, impedances = _make_dummy_spectrum()
        peak_idx = np.array([10, 11, 12, 25])
        val = loss.calculate(peak_log, peak_imp, freq_grid, impedances, peak_idx)
        assert val == 0

    def test_calculate_weight_when_no_split(self):
        loss = HarmonicSplittingLoss(harmonic_index=1, split_width_hz=2.0, weight=1.0)
        peak_log = np.log2(np.array([100.0, 200.0, 300.0]))
        peak_imp = np.ones(3) * 0.8
        freq_grid, impedances = _make_dummy_spectrum()
        peak_idx = np.array([10, 20, 30])
        val = loss.calculate(peak_log, peak_imp, freq_grid, impedances, peak_idx)
        assert val == 1.0

    def test_get_formula_returns_tuple(self):
        loss = HarmonicSplittingLoss(1, 5.0, 1.0)
        formula, symbols = loss.get_formula()
        assert isinstance(formula, str)
        assert isinstance(symbols, list)


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
