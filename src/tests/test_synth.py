import numpy as np
import pytest

from didgelab.synth import synthesize, Bore, ModelParams, Synthesizer, Controls, VOWELS
from didgelab.synth import vocaltract
from didgelab.synth.synth import _upsample
from didgelab.synth.techniques import parse_score, note_to_freq, score_controls
from didgelab.synth.excitation import lips_harmonics, lips_waveform

GEO = [[0, 32], [1500, 72]]


def test_vocal_tract_batch_equals_single():
    f = np.arange(20, 4000, 10.0)
    P = np.stack([v.as_array() for v in VOWELS.values()])
    Zb = vocaltract.impedance(P, np.tile(f, (len(P), 1)))
    for k, v in enumerate(VOWELS.values()):
        assert np.allclose(Zb[k], vocaltract.impedance(v, f))


def test_closure_raises_low_frequency_impedance():
    f = np.array([75.0, 150.0])
    open_ = np.abs(vocaltract.impedance(VOWELS["neutral"], f))
    closed = np.abs(vocaltract.impedance(VOWELS["neutral"].with_(tongue_height=1.0, tongue_pos=0.05), f))
    assert np.all(closed > 3 * open_)


def test_lip_source_power_law():
    U = np.abs(lips_harmonics(0.45, 0.85, 64))
    # slope of the harmonic envelope ~ n^-2 (-12 dB/octave) at high n
    slope = np.polyfit(np.log2(np.arange(16, 64)), 20 * np.log10(U[15:63]), 1)[0]
    assert -16 < slope < -8


def test_partial_bank_equals_direct_sum():
    b = Bore(GEO)
    p = ModelParams()
    c = Controls.constant(0.4, 500, b.drone, p, VOWELS["u"])
    c.amp *= np.linspace(0.5, 1.0, c.n)
    s = Synthesizer(b, p)
    s._prepare_vt(c)
    n = int(0.4 * 48000)
    y, theta = s.render_tone(c, n)
    A = s._harmonic_amplitudes(c, int(s.fmax / b.drone))
    y2 = sum(np.real(_upsample(A[:, k], n, 500, 48000) * np.exp(1j * (k + 1) * theta))
             for k in range(A.shape[1]))
    assert np.max(np.abs(y - y2)) < 1e-9 * np.max(np.abs(y2))


def test_score_parsing():
    ev, st = parse_score("tempo = 120\ndoo:2 di:.5 *too ~oo:1@A3 ^ rr - .")
    assert st["tempo"] == 120
    assert [e.kind for e in ev] == ["syllable"] * 4 + ["breath", "syllable", "rest"]
    assert ev[5].dur == pytest.approx(1.0)      # "rr" held for one more beat by "-"
    labels = [e.label for e in ev]
    assert labels[0] == "d+u" and labels[1] == "d+i"
    assert ev[2].toot == 1 and ev[3].voice == pytest.approx(220.0)
    assert ev[0].dur == pytest.approx(1.0)
    assert note_to_freq("A4") == pytest.approx(440.0)
    with pytest.raises(ValueError):
        parse_score("dxq")


def test_synthesize_runs_and_has_drone_pitch():
    y, fs = synthesize(GEO, "tempo = 120\ndoo:2 di:.5 ga:.5", reverb=0.0)
    assert np.all(np.isfinite(y)) and np.max(np.abs(y)) <= 1.0
    b = Bore(GEO)
    seg = y[int(0.3 * fs):int(0.9 * fs)]
    X = np.abs(np.fft.rfft(seg * np.hanning(len(seg)), 8 * len(seg)))
    f = np.fft.rfftfreq(8 * len(seg), 1 / fs)
    m = (f > 40) & (f < 1.5 * b.drone)
    f_peak = f[m][np.argmax(X[m])]
    assert abs(1200 * np.log2(f_peak / b.drone)) < 30


def test_toot_plays_second_resonance():
    b = Bore(GEO)
    c = score_controls(b, "tempo = 60\n*too:2", ModelParams())
    mid = c.f0[int(0.5 * c.rate):int(1.5 * c.rate)]
    target = b.toot(1) * 2 ** (ModelParams().toot_cents / 1200)
    assert abs(1200 * np.log2(np.median(mid) / target)) < 15


def test_bore_inertia_time_constant():
    b = Bore(GEO)
    s = Synthesizer(b, ModelParams())
    tau_drone = s.envelope_time_constant(b.drone)
    q_based = b.resonances[0].q / (np.pi * b.drone)
    assert 0.5 * q_based < tau_drone < 1.5 * q_based
    # between two resonances the bore responds quickly
    assert s.envelope_time_constant(0.5 * (b.resonances[0].freq + b.resonances[1].freq)) < 0.02
