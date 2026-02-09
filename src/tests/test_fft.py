"""
Pytest unit tests for didgelab.fft.
"""

import tempfile
import os
import pytest
import numpy as np
from scipy.io import wavfile

from didgelab.fft import (
    do_fft,
    get_harmonic_maxima,
    sling_window_average_spectrum,
    get_fundamental,
    get_peaks,
)


def _make_test_wav(filepath, freq_hz=440, duration_s=0.5, sample_rate=44100, stereo=False):
    """Create a mono or stereo WAV file with a sine tone."""
    t = np.linspace(0, duration_s, int(sample_rate * duration_s), dtype=np.float32)
    signal = (np.sin(2 * np.pi * freq_hz * t) * 32767).astype(np.int16)
    if stereo:
        signal = np.column_stack([signal, signal])
    wavfile.write(filepath, sample_rate, signal)


class TestDoFft:
    """Tests for do_fft."""

    def test_mono_wav_returns_freq_and_magnitude(self):
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            path = f.name
        try:
            _make_test_wav(path, freq_hz=440, duration_s=0.1, stereo=False)
            freq, mag = do_fft(path, maxfreq=1000)
            assert isinstance(freq, np.ndarray)
            assert isinstance(mag, np.ndarray)
            assert len(freq) == len(mag)
            assert freq[-1] <= 1000
            assert freq[0] >= 0
        finally:
            os.unlink(path)

    def test_stereo_wav_uses_left_channel(self):
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            path = f.name
        try:
            _make_test_wav(path, freq_hz=440, duration_s=0.1, stereo=True)
            freq, mag = do_fft(path, maxfreq=1000)
            assert len(freq) == len(mag)
            assert freq[-1] <= 1000
        finally:
            os.unlink(path)

    def test_maxfreq_cuts_spectrum(self):
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            path = f.name
        try:
            _make_test_wav(path, freq_hz=200, duration_s=0.1)
            freq_500, _ = do_fft(path, maxfreq=500)
            freq_2000, _ = do_fft(path, maxfreq=2000)
            assert freq_500[-1] <= 500
            assert freq_2000[-1] <= 2000
            assert len(freq_2000) >= len(freq_500)
        finally:
            os.unlink(path)


class TestGetHarmonicMaxima:
    """Tests for get_harmonic_maxima."""

    def test_synthetic_harmonics(self):
        # Synthetic spectrum with single strong peak at 100 Hz (fundamental)
        # Algorithm finds first max above min_freq as base, then harmonics
        freq = np.linspace(50, 1000, 1000)
        spectrum = np.exp(-((freq - 100) ** 2) / 50) + 0.5 * np.exp(
            -((freq - 200) ** 2) / 50
        )
        maxima = get_harmonic_maxima(freq, spectrum, min_freq=60)
        assert len(maxima) >= 1
        assert maxima[0] == pytest.approx(100, abs=15)
        assert all(50 < m < 1050 for m in maxima)

    def test_empty_spectrum_returns_empty_list(self):
        freq = np.array([10, 20])
        spectrum = np.array([0.0, 0.0])
        maxima = get_harmonic_maxima(freq, spectrum, min_freq=60)
        assert maxima == []


class TestSlingWindowAverageSpectrum:
    """Tests for sling_window_average_spectrum (note: typo in function name)."""

    def test_downsample_reduces_length(self):
        freq = np.arange(100, dtype=float)
        spectrum = np.ones(100)
        new_freq, new_spec = sling_window_average_spectrum(freq, spectrum, window_size=5)
        assert len(new_freq) < len(freq)
        assert len(new_spec) == len(new_freq)

    def test_window_size_5_averages_5_points(self):
        freq = np.arange(20, dtype=float)
        spectrum = np.arange(20, dtype=float)  # 0..19
        new_freq, new_spec = sling_window_average_spectrum(freq, spectrum, window_size=5)
        # First window: indices 0..4, mean = 2.0
        assert new_spec[0] == pytest.approx(2.0)

    def test_output_arrays_match_length(self):
        freq = np.arange(50, dtype=float)
        spectrum = np.random.rand(50)
        new_freq, new_spec = sling_window_average_spectrum(freq, spectrum, window_size=5)
        assert len(new_freq) == len(new_spec)


class TestGetFundamental:
    """Tests for get_fundamental."""

    def test_single_peak_in_range(self):
        freq = np.linspace(1, 200, 500)
        spectrum = np.exp(-((freq - 73) ** 2) / 50)  # peak near 73 Hz
        f, idx = get_fundamental(freq, spectrum, minfreq=50, maxfreq=120, order=40)
        assert 50 <= f <= 120
        assert idx >= 0
        assert freq[idx] == pytest.approx(f)

    def test_assert_single_local_max_in_range(self):
        # Multiple local maxima in range -> assert fails
        freq = np.linspace(1, 200, 500)
        spectrum = np.exp(-((freq - 70) ** 2) / 50) + 0.5 * np.exp(-((freq - 90) ** 2) / 50)
        with pytest.raises(AssertionError):
            get_fundamental(freq, spectrum, minfreq=50, maxfreq=120, order=10)


class TestGetPeaks:
    """Tests for get_peaks."""

    def test_includes_fundamental_and_higher_peaks(self):
        # Spectrum with clear fundamental at ~73 Hz and harmonic at ~146 Hz
        freq = np.linspace(1, 250, 1000)
        spectrum = np.exp(-((freq - 73) ** 2) / 100) + 0.8 * np.exp(-((freq - 146) ** 2) / 100)
        peaks = get_peaks(freq, spectrum, return_indizes=False)
        assert isinstance(peaks, np.ndarray)
        assert len(peaks) >= 1
        assert peaks[0] == pytest.approx(73, abs=15)

    def test_return_indizes(self):
        freq = np.linspace(1, 250, 1000)
        spectrum = np.exp(-((freq - 73) ** 2) / 100)
        peaks, indizes = get_peaks(freq, spectrum, return_indizes=True)
        assert len(indizes) == len(peaks)
        for i, idx in enumerate(indizes):
            assert 0 <= idx < len(freq)
            assert freq[idx] == pytest.approx(peaks[i])
