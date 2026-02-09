"""
Musical note and frequency conversion utilities for DidgeLab.

Provides conversion between note numbers, note names (e.g. A4), frequencies (Hz),
wavelengths, and cent differences. Used for tuning targets and interpreting
impedance spectra (e.g. mapping peaks to notes).

All functions accept "dual parameters": arguments can be either Python scalars
(int, float) or numpy arrays; the return type matches (scalar in -> scalar out,
array in -> array out). Broadcasting applies when multiple array arguments are used.
"""

import math
import numpy as np


def _is_scalar(x):
    """True if x is a Python scalar or 0-d array (single value)."""
    if isinstance(x, (int, float)):
        return True
    if isinstance(x, np.ndarray) and x.ndim == 0:
        return True
    return False


def _as_array(x):
    """Convert to numpy array (at least 1-d) for vectorized computation."""
    return np.atleast_1d(np.asarray(x, dtype=np.float64))


def note_to_freq(note, base_freq=440):
    """Convert note number to frequency in Hz (e.g. 0 -> base_freq, 12 -> 2*base_freq).
    Accepts scalar or array for note and base_freq; broadcasts."""
    note_scalar = _is_scalar(note)
    base_scalar = _is_scalar(base_freq)
    n = _as_array(note)
    b = _as_array(np.asarray(base_freq, dtype=np.float64))
    out = np.broadcast_to(b * np.power(2, n / 12), np.broadcast_shapes(n.shape, b.shape))
    if note_scalar and base_scalar:
        return float(out.flat[0])
    return out


def note_name(note):
    """Convert note number to note name (e.g. 0 -> A4). Accepts scalar or array."""
    note_scalar = _is_scalar(note)
    n = _as_array(note)
    # Round and compute octave/number per element
    note_int = np.round(n).astype(int)
    note_int = note_int + 48
    octave = np.floor((note_int - 3) / 12).astype(int) + 1
    number = note_int % 12
    names = ["A", "A#", "B", "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#"]
    result = np.array([str(names[num]) + str(oct) for num, oct in zip(number.flat, octave.flat)])
    result = result.reshape(n.shape)
    if note_scalar:
        return result.flat[0]
    return result


def freq_to_note(freq, base_freq=440):
    """Convert frequency in Hz to note number relative to base_freq (e.g. 440 -> 0).
    Accepts scalar or array for freq and base_freq; broadcasts."""
    freq_scalar = _is_scalar(freq)
    base_scalar = _is_scalar(base_freq)
    f = _as_array(freq)
    b = _as_array(np.asarray(base_freq, dtype=np.float64))
    out = np.broadcast_to(12 * (np.log2(f) - np.log2(b)), np.broadcast_shapes(f.shape, b.shape))
    if freq_scalar and base_scalar:
        return float(out.flat[0])
    return out


def freq_to_note_and_cent(freq, base_freq=440):
    """Return (note_number, cent_diff) for a frequency relative to base_freq.
    Accepts scalar or array for freq and base_freq; returns tuples of arrays or scalars."""
    freq_scalar = _is_scalar(freq)
    base_scalar = _is_scalar(base_freq)
    f = _as_array(freq)
    b = _as_array(np.asarray(base_freq, dtype=np.float64))
    note_fuzzy = 12 * (np.log2(f) - np.log2(b))
    note_int = np.round(note_fuzzy).astype(int)
    diff = note_fuzzy - note_int
    cents = diff * 100
    if freq_scalar and base_scalar:
        return int(note_int.flat[0]), float(cents.flat[0])
    return note_int, cents


def freq_to_wavelength(freq):
    """Wavelength of sound at given frequency in mm (c=343.2 m/s). Accepts scalar or array."""
    freq_scalar = _is_scalar(freq)
    f = _as_array(freq)
    c = 343.2
    out = 1000 * c / f
    if freq_scalar:
        return float(out.flat[0])
    return out


def note_name_to_number(note):
    """Convert note name to note number (e.g. A4 -> 0 for A4 with base 440).
    Accepts a single string or array-like of strings."""
    names = ["A", "A#", "B", "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#"]

    def _one(s):
        assert len(s) in (2, 3)
        if len(s) == 3:
            assert s[1] == "#"
        octave = int(s[-1])
        n = names.index(s[0 : len(s) - 1])
        return 12 * octave + n - 48

    if isinstance(note, str):
        return _one(note)
    note_arr = np.atleast_1d(note)
    if note_arr.ndim != 1:
        raise ValueError("note_name_to_number expects a string or 1-d array of strings")
    out = np.array([_one(str(s)) for s in note_arr.flat], dtype=np.int_)
    out = out.reshape(note_arr.shape)
    if note_arr.size == 1:
        return int(out.flat[0])
    return out


def cent_diff(freq1, freq2):
    """Difference between two frequencies in cents (1200 * log2(freq2/freq1)).
    Accepts scalar or array for freq1 and freq2; broadcasts."""
    f1_scalar = _is_scalar(freq1)
    f2_scalar = _is_scalar(freq2)
    f1 = _as_array(freq1)
    f2 = _as_array(freq2)
    out = 1200 * np.log2(f2 / f1)
    if f1_scalar and f2_scalar:
        return float(out.flat[0])
    return out
