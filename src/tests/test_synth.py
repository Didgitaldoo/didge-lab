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


def test_inertia_equals_resonance_bandwidth():
    # tau = 1 / (pi B) with B the exact -3 dB bandwidth of the drone peak
    from didgelab.synth import ModalBore
    from didgelab.synth.bore import chain_matrix
    b = Bore(GEO)
    f = np.arange(b.drone - 6, b.drone + 6, 0.002)
    A, B, C, D, Zr = chain_matrix(b.geo, f, b.constants)
    h = np.abs(1 / (C * Zr + D))
    above = f[h > h.max() / np.sqrt(2)]
    tau = Synthesizer(b, ModelParams()).envelope_time_constant(f[np.argmax(h)])
    assert abs(tau * np.pi * (above[-1] - above[0]) - 1) < 0.03
    # a single modal resonance: tau = Q / (pi f)
    m = ModalBore([(70.0, 2e7, 30.0), (200.0, 8e6, 30.0)])
    assert abs(Synthesizer(m, ModelParams()).envelope_time_constant(70.0) * np.pi * 70.0 / 30.0 - 1) < 0.02


def test_high_frequencies_are_filled_by_the_broadband_noise():
    b = Bore(GEO)
    c = score_controls(b, "tempo = 60\nuh:2", ModelParams())

    def band_db(params):
        y = Synthesizer(b, params, seed=0).render(c, reverb=0.0, normalize=False)
        X = np.abs(np.fft.rfft(y)) ** 2
        f = np.fft.rfftfreq(len(y), 1 / 48000)
        lo = X[(f > 125) & (f < 1000)].sum()
        return 10 * np.log10(X[(f > 12000) & (f < 16000)].sum() / lo)

    with_noise = band_db(ModelParams().with_(hf_noise_db=-20))   # off by default
    without = band_db(ModelParams().with_(hf_noise_db=-200, noise_db=-200))
    assert with_noise > without + 15          # the tone stops at 12 kHz (leakage: ~-53 dB)
    assert -70 < with_noise < -20


def test_very_low_drone_renders():
    b = Bore([[0, 30], [3000, 32]])
    assert b.drone < 40
    y, fs = synthesize([[0, 30], [3000, 32]], "tempo = 60\ndoo:1", reverb=0.0)
    assert np.isfinite(y).all() and np.abs(y).max() > 0


def test_articulated_playing_keeps_the_mids():
    # Didge Archive songs: 300 Hz - 1 kHz carries about as much of the energy as
    # in the neutral drone; a nearly closed jaw made the articulated shapes 7 dB darker
    b = Bore(GEO)

    def share(score):
        c = score_controls(b, score, ModelParams())
        y = Synthesizer(b, ModelParams(), seed=0).render(c, reverb=0.0)
        X = np.abs(np.fft.rfft(y)) ** 2
        f = np.fft.rfftfreq(len(y), 1 / 48000)
        return 10 * np.log10(X[(f >= 300) & (f < 1000)].sum() / X[(f >= 40) & (f < 20000)].sum())

    drone = share("tempo = 60\nuh:3")
    groove = share("tempo = 100\ndi:.5 ga:.5 ti:.5 ka:.5 di:.5 ga:.5 ti:.5 ka:.5 ee:1 oo:1")
    assert groove > drone - 4


def test_low_open_quotient_reaches_the_sound():
    # the player used to clamp oq to >= 0.2: lower values sounded the same
    b = Bore(GEO)

    def centroid(oq):
        p = ModelParams(oq=oq)
        c = score_controls(b, "tempo = 60\nuh:2", p)
        assert abs(np.median(c.oq) - oq) < 1e-9
        y = Synthesizer(b, p, seed=0).render(c, reverb=0.0)
        X = np.abs(np.fft.rfft(y)) ** 2
        f = np.fft.rfftfreq(len(y), 1 / 48000)
        return (f * X).sum() / X.sum()

    assert centroid(0.1) > 1.3 * centroid(0.2)
