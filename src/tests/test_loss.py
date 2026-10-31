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
try:
    import didgelab.sim.tlm_cython_lib._cadsd  # use the real extension when it is built
except ImportError:
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
    half_power_q,
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


def _make_impedance_with_peaks(peak_freqs_hz, q=30.0):
    """Return a fake acoustical_simulation(geo, freqs) with resonance peaks at the given frequencies.

    Each peak is a resonance curve of quality factor q on a 1e5 floor, so the result is
    valid for any frequency grid (the loss simulates a coarse grid plus fine peak windows).
    """
    def simulate(geo, freqs):
        freqs = np.asarray(freqs, dtype=float)
        impedances = np.ones(len(freqs)) * 1e5
        for f_hz in peak_freqs_hz:
            impedances += 1e7 / np.sqrt(1 + (2 * q * (freqs / f_hz - 1)) ** 2)
        return impedances
    return simulate


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

    def test_p1_matches_uniform(self):
        """p=1 is neutral: same as default uniform weighting."""
        peak_log = np.log2(np.array([80.0, 160.0, 240.0]))
        peak_imp = np.ones(3) * 0.8
        freq_grid, impedances = _make_dummy_spectrum()
        peak_idx = np.array([10, 30, 50])
        uniform = ScaleTuningLoss(60, [0, 2, 4, 5, 7, 9, 11], 1.0)
        neutral = ScaleTuningLoss(60, [0, 2, 4, 5, 7, 9, 11], 1.0, favor_lower_frequencies=1.0)
        assert uniform.calculate(peak_log, peak_imp, freq_grid, impedances, peak_idx) == pytest.approx(
            neutral.calculate(peak_log, peak_imp, freq_grid, impedances, peak_idx)
        )

    def test_includes_notes_below_c4(self):
        """Didgeridoo drones sit below C4; the scale grid must include those octaves."""
        # Same mapping as evolution_runner: E → MIDI 64 (E4), major pentatonic.
        loss = ScaleTuningLoss(64, [0, 2, 4, 7, 9], 1.0)
        freqs = np.power(2.0, loss.scale_freqs_log)
        assert freqs.min() < 40.0
        assert freqs.max() <= 1000.0

        def midi_hz(m: int) -> float:
            return 440.0 * (2.0 ** ((m - 69.0) / 12.0))

        e2 = midi_hz(40)
        fs3 = midi_hz(54)
        e4 = midi_hz(64)
        for target in (e2, fs3, e4):
            cents = np.min(np.abs(1200.0 * np.log2(freqs / target)))
            assert cents == pytest.approx(0.0, abs=1e-6)

    def test_in_scale_drone_has_near_zero_loss(self):
        e2 = 440.0 * (2.0 ** ((40 - 69.0) / 12.0))
        loss = ScaleTuningLoss(64, [0, 2, 4, 7, 9], 1.0)
        peak_log = np.log2(np.array([e2]))
        peak_imp = np.array([1.0])
        freq_grid, impedances = _make_dummy_spectrum()
        peak_idx = np.array([10])
        val = loss.calculate(peak_log, peak_imp, freq_grid, impedances, peak_idx)
        assert val == pytest.approx(0.0, abs=1e-9)

    def test_p_gt_1_favors_lower(self):
        """With only the low peak mistuned, p>1 yields higher loss than p=1."""
        # High peak on a scale note (C5 ≈ 523.25 from C4 major); low peak offset.
        c5 = 440.0 * (2.0 ** ((72 - 69) / 12.0))
        off_low = 70.0  # between C2 (65.4 Hz) and D2 (73.4 Hz)
        peak_log = np.log2(np.array([off_low, c5]))
        peak_imp = np.ones(2)
        freq_grid, impedances = _make_dummy_spectrum()
        peak_idx = np.array([5, 40])
        loss_p1 = ScaleTuningLoss(60, [0, 2, 4, 5, 7, 9, 11], 1.0, favor_lower_frequencies=1.0)
        loss_p2 = ScaleTuningLoss(60, [0, 2, 4, 5, 7, 9, 11], 1.0, favor_lower_frequencies=2.0)
        v1 = loss_p1.calculate(peak_log, peak_imp, freq_grid, impedances, peak_idx)
        v2 = loss_p2.calculate(peak_log, peak_imp, freq_grid, impedances, peak_idx)
        assert v2 > v1

    def test_p_lt_1_favors_higher(self):
        """With only the high peak mistuned, p<1 yields higher loss than p=1."""
        c4 = 440.0 * (2.0 ** ((60 - 69) / 12.0))
        off_high = 400.0
        peak_log = np.log2(np.array([c4, off_high]))
        peak_imp = np.ones(2)
        freq_grid, impedances = _make_dummy_spectrum()
        peak_idx = np.array([10, 50])
        loss_p1 = ScaleTuningLoss(60, [0, 2, 4, 5, 7, 9, 11], 1.0, favor_lower_frequencies=1.0)
        loss_p0 = ScaleTuningLoss(60, [0, 2, 4, 5, 7, 9, 11], 1.0, favor_lower_frequencies=0.0)
        v1 = loss_p1.calculate(peak_log, peak_imp, freq_grid, impedances, peak_idx)
        v0 = loss_p0.calculate(peak_log, peak_imp, freq_grid, impedances, peak_idx)
        assert v0 > v1

    def test_identical_distances_independent_of_p(self):
        """When every peak has the same cents error, loss does not depend on p."""
        # Two peaks offset by the same log2 amount from their nearest scale notes.
        # Use frequencies that share the same min cents distance by construction:
        # pick one scale note and offset both peaks by the same delta in log2.
        intervals = [0, 2, 4, 5, 7, 9, 11]
        base = ScaleTuningLoss(60, intervals, 1.0)
        f0 = 2.0 ** base.scale_freqs_log[0]
        f1 = 2.0 ** base.scale_freqs_log[7]  # one octave up in the generated grid
        delta_log = 0.05  # same offset for both
        peak_log = np.array([np.log2(f0) + delta_log, np.log2(f1) + delta_log])
        peak_imp = np.ones(2)
        freq_grid, impedances = _make_dummy_spectrum()
        peak_idx = np.array([10, 30])
        vals = [
            ScaleTuningLoss(60, intervals, 1.0, favor_lower_frequencies=p).calculate(
                peak_log, peak_imp, freq_grid, impedances, peak_idx
            )
            for p in (0.0, 1.0, 2.0, 3.0)
        ]
        assert vals[0] == pytest.approx(vals[1], rel=1e-9)
        assert vals[1] == pytest.approx(vals[2], rel=1e-9)
        assert vals[2] == pytest.approx(vals[3], rel=1e-9)


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

    @pytest.mark.parametrize("q", [10.0, 36.0, 80.0])
    def test_half_power_q_on_known_resonances(self, q):
        """Direct call: Q is measured from the half-power crossings of the given spectrum.
        One isolated peak, so the resonance curve is not widened by its neighbours."""
        from didgelab.acoustical_simulation import get_log_simulation_frequencies

        peaks_hz = [150.0]
        freqs = get_log_simulation_frequencies(30, 1000, 1.0)
        impedances = _make_impedance_with_peaks(peaks_hz, q=q)(None, freqs)
        peak_idx = np.array([np.argmax(np.where(np.abs(freqs / f - 1) < 0.05, impedances, 0)) for f in peaks_hz])

        measured = half_power_q(freqs, impedances, peak_idx)
        np.testing.assert_allclose(measured, q, rtol=0.05)
        loss = QFactorLoss(target_q=q, weight=1.0)
        assert loss.calculate(np.log2(peaks_hz), np.ones(1), freqs, impedances, peak_idx) < 0.05 * q

    @pytest.mark.parametrize("q", [10.0, 36.0, 80.0])
    def test_composite_q_from_refined_peaks(self, q):
        """Inside CompositeTairuaLoss the Q of the refined peaks is accurate on the coarse grid."""
        peaks_hz = [150.0]
        composite = CompositeTairuaLoss(max_error=5.0)
        composite.add_component("q_factor", QFactorLoss(target_q=q, weight=1.0))
        shape = KigaliShape(n_segments=20)
        shape.loss = None

        with patch("didgelab.loss.loss.acoustical_simulation", side_effect=_make_impedance_with_peaks(peaks_hz, q=q)):
            result = composite.loss(shape)

        assert result["q_factor"] < 0.05 * q

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

        simulate = _make_impedance_with_peaks([73.4, 146.8])

        with patch("didgelab.loss.loss.acoustical_simulation", side_effect=simulate):
            result = loss.loss(shape)

        assert "total" in result, "Result must contain 'total' key"
        for key in EXPECTED_COMPONENT_KEYS:
            assert key in result, f"Result must contain component key '{key}'"

        assert len(result) == len(EXPECTED_COMPONENT_KEYS) + 1
        assert isinstance(result["total"], (int, float))
        assert result["total"] >= 0

    def test_refined_peaks_are_sub_cent_accurate(self):
        """Peaks found on the coarse grid are refined to well below the max_error resolution."""
        loss = CompositeTairuaLoss(max_error=5.0)
        peaks_hz = [61.7, 173.3, 291.9]
        loss.add_component("freq", FrequencyTuningLoss(np.log2(peaks_hz), np.full(3, -1.0), [1.0, 1.0, 1.0]))
        shape = KigaliShape(n_segments=20)
        shape.loss = None

        with patch("didgelab.loss.loss.acoustical_simulation", side_effect=_make_impedance_with_peaks(peaks_hz)):
            result = loss.loss(shape)

        # freq loss is sum(cents / 600); allow 0.5 cent per peak
        assert result["freq"] < 3 * 0.5 / 600


class TestResiduals:
    """Signed residuals for least-squares refinement: sum(|r|) must equal the loss."""

    def test_frequency_tuning_residuals_are_signed_and_sum_to_loss(self):
        targets = np.log2(np.array([80.0, 160.0]))
        loss = FrequencyTuningLoss(targets, np.array([-1.0, 0.5]), [2.0, 1.0])
        peaks = targets + np.array([10.0, -20.0]) / 1200
        imps = np.array([0.9, 0.3])

        r = loss.residuals(peaks, imps, None, None, None)

        np.testing.assert_allclose(r, [2.0 * 10 / 600, 1.0 * -20 / 600, 1.0 * (0.3 - 0.5)])
        assert np.sum(np.abs(r)) == pytest.approx(loss.calculate(peaks, imps, None, None, None))

    @pytest.mark.parametrize("favor_lower", [1.0, 1.5])
    def test_scale_tuning_residuals_sum_to_loss(self, favor_lower):
        loss = ScaleTuningLoss(base_note=60, intervals=[0, 4, 7], weight=3.0, favor_lower_frequencies=favor_lower)
        peaks = np.log2(np.array([65.0, 98.0, 131.5, 200.0]))

        r = loss.residuals(peaks, np.ones(4), None, None, None)

        assert len(r) == 4
        assert np.sum(np.abs(r)) == pytest.approx(loss.calculate(peaks, np.ones(4), None, None, None))
        assert np.any(r < 0) and np.any(r > 0)

    def test_components_without_residuals_return_none(self):
        assert PeakQuantityLoss(4, 1.0).residuals(np.zeros(2), np.ones(2), None, None, None) is None

    def test_composite_residuals(self):
        loss = CompositeTairuaLoss(max_error=5.0)
        loss.add_component("freq", FrequencyTuningLoss(np.log2([72.0, 150.0]), np.full(2, -1.0), [1.0, 1.0]))
        loss.add_component("peaks_qty", PeakQuantityLoss(target_count=5, weight=2.0))
        shape = KigaliShape(n_segments=20)
        shape.loss = None

        with patch("didgelab.loss.loss.acoustical_simulation", side_effect=_make_impedance_with_peaks([73.4, 146.8])):
            r, results = loss.residuals(shape)
            assert results == pytest.approx(loss.loss(shape))

        # two signed tuning residuals, then sqrt of the peak-quantity loss (no residuals)
        assert len(r) == 3
        assert r[0] > 0 and r[1] < 0  # peak 73.4 Hz above target 72, 146.8 Hz below 150
        assert r[2] ** 2 == pytest.approx(results["peaks_qty"])
        assert np.sum(np.abs(r[:2])) == pytest.approx(results["freq"])
