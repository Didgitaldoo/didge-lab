"""
Pytest unit tests for didgelab.conv (note/frequency conversion utilities).
"""

import math
import pytest
import numpy as np

from didgelab.conv import (
    note_to_freq,
    note_name,
    freq_to_note,
    freq_to_note_and_cent,
    freq_to_wavelength,
    note_name_to_number,
    cent_diff,
)


class TestNoteToFreq:
    """Tests for note_to_freq."""

    def test_zero_note_returns_base_freq(self):
        assert note_to_freq(0, 440) == 440.0

    def test_default_base_freq_440(self):
        assert note_to_freq(0) == 440.0

    def test_one_octave_up_doubles_freq(self):
        assert note_to_freq(12, 440) == 880.0

    def test_one_octave_down_halves_freq(self):
        assert note_to_freq(-12, 440) == 220.0

    def test_custom_base_freq(self):
        assert note_to_freq(0, 432) == 432.0
        assert note_to_freq(12, 432) == 864.0

    def test_semitone_ratio(self):
        # 1 semitone = 2^(1/12)
        expected = 440 * (2 ** (1 / 12))
        assert note_to_freq(1, 440) == pytest.approx(expected)


class TestNoteName:
    """Tests for note_name."""

    def test_a4_at_note_zero(self):
        assert note_name(0) == "A4"

    def test_rounds_before_convert(self):
        assert note_name(0.4) == "A4"
        assert note_name(0.6) == "A#4"

    def test_common_notes(self):
        # Note numbers relative to A4=0: C4 = -9, G4 = -2, etc.
        assert note_name(-9) == "C4"
        assert note_name(2) == "B4"
        assert note_name(3) == "C5"

    def test_sharps(self):
        assert note_name(1) == "A#4"
        assert note_name(-8) == "C#4"


class TestFreqToNote:
    """Tests for freq_to_note."""

    def test_base_freq_is_zero(self):
        assert freq_to_note(440, 440) == pytest.approx(0.0)

    def test_double_freq_is_twelve(self):
        assert freq_to_note(880, 440) == pytest.approx(12.0)

    def test_half_freq_is_minus_twelve(self):
        assert freq_to_note(220, 440) == pytest.approx(-12.0)

    def test_custom_base(self):
        assert freq_to_note(432, 432) == pytest.approx(0.0)


class TestFreqToNoteAndCent:
    """Tests for freq_to_note_and_cent."""

    def test_exact_freq_gives_zero_cents(self):
        note, cents = freq_to_note_and_cent(440, 440)
        assert note == 0
        assert cents == pytest.approx(0.0, abs=1e-9)

    def test_slightly_flat_negative_cents(self):
        # Slightly below 440 -> negative cent diff from A4
        note, cents = freq_to_note_and_cent(438, 440)
        assert note == 0
        assert cents < 0

    def test_one_octave_up(self):
        note, cents = freq_to_note_and_cent(880, 440)
        assert note == 12
        assert cents == pytest.approx(0.0, abs=1e-9)


class TestFreqToWavelength:
    """Tests for freq_to_wavelength."""

    def test_wavelength_at_speed_of_sound(self):
        # At 343.2 Hz, wavelength = 1000 mm = 1 m
        assert freq_to_wavelength(343.2) == pytest.approx(1000.0)

    def test_higher_freq_shorter_wavelength(self):
        assert freq_to_wavelength(686.4) == pytest.approx(500.0)

    def test_lower_freq_longer_wavelength(self):
        assert freq_to_wavelength(100) == pytest.approx(3432.0)


class TestNoteNameToNumber:
    """Tests for note_name_to_number."""

    def test_a4_is_zero(self):
        assert note_name_to_number("A4") == 0

    def test_c4(self):
        # In this scheme: A4=0, A#4=1, B4=2, C4=3 (12*octave + pitch_index - 48)
        assert note_name_to_number("C4") == 3

    def test_sharp_notes(self):
        assert note_name_to_number("A#4") == 1
        # F# is index 9 in names; 12*4+9-48 = 9
        assert note_name_to_number("F#4") == 9

    def test_other_octaves(self):
        assert note_name_to_number("A5") == 12
        assert note_name_to_number("A3") == -12

    def test_invalid_length_raises(self):
        with pytest.raises(AssertionError):
            note_name_to_number("A")
        with pytest.raises(AssertionError):
            note_name_to_number("A44")


class TestCentDiff:
    """Tests for cent_diff."""

    def test_same_freq_zero_cents(self):
        assert cent_diff(440, 440) == pytest.approx(0.0)

    def test_one_octave_1200_cents(self):
        assert cent_diff(440, 880) == pytest.approx(1200.0)

    def test_half_octave_600_cents(self):
        assert cent_diff(440, 440 * (2 ** 0.5)) == pytest.approx(600.0)

    def test_one_semitone_100_cents(self):
        assert cent_diff(440, 440 * (2 ** (1 / 12))) == pytest.approx(100.0)

    def test_numpy_inputs(self):
        a = np.array([440, 880])
        b = np.array([880, 440])
        result = cent_diff(a, b)
        assert result[0] == pytest.approx(1200.0)
        assert result[1] == pytest.approx(-1200.0)


class TestRoundTrip:
    """Round-trip and consistency between conv functions."""

    def test_note_to_freq_to_note(self):
        for note in [-12, 0, 1, 12, 24]:
            f = note_to_freq(note, 440)
            back = freq_to_note(f, 440)
            assert back == pytest.approx(note, abs=1e-10)

    def test_note_name_to_number_to_note_name(self):
        # Round-trip for note names that use the same octave convention as note_name
        for name in ["A4", "A5", "A3", "A#4", "A#5", "B4"]:
            num = note_name_to_number(name)
            back = note_name(num)
            assert back == name

    def test_freq_to_note_and_cent_consistent_with_freq_to_note(self):
        freq = 466.16  # A#4
        note_f, cent = freq_to_note_and_cent(freq, 440)
        note_direct = freq_to_note(freq, 440)
        assert note_f == round(note_direct)
        assert (note_f + cent / 100) == pytest.approx(note_direct, abs=1e-10)


class TestDualParameters:
    """Tests that functions accept both scalars and numpy arrays (dual parameters)."""

    def test_note_to_freq_array(self):
        notes = np.array([0, 12, -12])
        out = note_to_freq(notes, 440)
        assert isinstance(out, np.ndarray)
        assert out.shape == (3,)
        assert out[0] == pytest.approx(440.0)
        assert out[1] == pytest.approx(880.0)
        assert out[2] == pytest.approx(220.0)

    def test_note_to_freq_broadcast(self):
        notes = np.array([0, 12])
        bases = np.array([440, 432])
        out = note_to_freq(notes, bases)
        assert out.shape == (2,)
        assert out[0] == pytest.approx(440.0)
        assert out[1] == pytest.approx(864.0)

    def test_note_name_array(self):
        notes = np.array([0, 12, 3])
        out = note_name(notes)
        assert isinstance(out, np.ndarray)
        assert out.shape == (3,)
        assert out[0] == "A4"
        assert out[1] == "A5"
        assert out[2] == "C5"

    def test_freq_to_note_array(self):
        freqs = np.array([440, 880, 220])
        out = freq_to_note(freqs, 440)
        assert isinstance(out, np.ndarray)
        assert out.shape == (3,)
        assert out[0] == pytest.approx(0.0)
        assert out[1] == pytest.approx(12.0)
        assert out[2] == pytest.approx(-12.0)

    def test_freq_to_note_and_cent_array(self):
        freqs = np.array([440, 880])
        notes, cents = freq_to_note_and_cent(freqs, 440)
        assert isinstance(notes, np.ndarray)
        assert isinstance(cents, np.ndarray)
        assert notes.shape == (2,)
        assert notes[0] == 0 and notes[1] == 12
        assert cents[0] == pytest.approx(0.0)
        assert cents[1] == pytest.approx(0.0)

    def test_freq_to_wavelength_array(self):
        freqs = np.array([343.2, 686.4])
        out = freq_to_wavelength(freqs)
        assert isinstance(out, np.ndarray)
        assert out.shape == (2,)
        assert out[0] == pytest.approx(1000.0)
        assert out[1] == pytest.approx(500.0)

    def test_note_name_to_number_array(self):
        names = np.array(["A4", "A#4", "C4"])
        out = note_name_to_number(names)
        assert isinstance(out, np.ndarray)
        assert out.shape == (3,)
        assert out[0] == 0
        assert out[1] == 1
        assert out[2] == 3

    def test_note_name_to_number_list(self):
        names = ["A4", "A5"]
        out = note_name_to_number(names)
        assert isinstance(out, np.ndarray)
        assert out[0] == 0 and out[1] == 12

    def test_cent_diff_scalar_out_when_both_scalar(self):
        out = cent_diff(440, 880)
        assert isinstance(out, float)
        assert out == pytest.approx(1200.0)

    def test_cent_diff_array_out_when_any_array(self):
        out = cent_diff(np.array([440, 220]), np.array([880, 440]))
        assert isinstance(out, np.ndarray)
        assert out.shape == (2,)
        assert out[0] == pytest.approx(1200.0)
        assert out[1] == pytest.approx(1200.0)
