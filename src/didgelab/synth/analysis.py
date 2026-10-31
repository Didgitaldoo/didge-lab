"""
Signal analysis of didgeridoo recordings (used by ``didgelab.synth.impedance``).

* robust f0 tracking (harmonic-contrast score, refined by parabolic interpolation)
* harmonic amplitudes per frame

Ported from didge-synth (``didgesynth.analysis``).
"""

from dataclasses import dataclass

import numpy as np


def stft(x, fs, n_fft=8192, win=4096, hop=480):
    """Magnitude STFT, returns (times, freqs, |X|)."""
    w = np.hanning(win)
    pad = np.concatenate([np.zeros(win // 2), x, np.zeros(win)])
    n_frames = max(1, (len(pad) - win) // hop)
    idx = np.arange(win)[None, :] + hop * np.arange(n_frames)[:, None]
    frames = pad[idx] * w[None, :]
    X = np.abs(np.fft.rfft(frames, n_fft, axis=1)) / (w.sum() / 2)
    t = np.arange(n_frames) * hop / fs
    f = np.fft.rfftfreq(n_fft, 1 / fs)
    return t, f, X


def _sample(S, f, freqs):
    """Linear interpolation of the log spectrum S (frames x bins) at freqs (frames x k)."""
    df = f[1] - f[0]
    pos = np.clip(freqs / df, 0, S.shape[1] - 2)
    i = pos.astype(int)
    a = pos - i
    rows = np.arange(S.shape[0])[:, None]
    return S[rows, i] * (1 - a) + S[rows, i + 1] * a


@dataclass
class F0Track:
    t: np.ndarray
    f0: np.ndarray          # Hz, NaN where unvoiced/silent
    score: np.ndarray       # harmonic contrast in dB
    rms_db: np.ndarray


def track_f0(x, fs, fmin=40.0, fmax=500.0, n_harm=12, hop=480, min_score=6.0,
             spec=None, sub_margin=8.0):
    """Harmonic-contrast pitch tracker.

    For a candidate f the score is the mean log magnitude at the harmonics n*f
    minus the mean log magnitude at the half-way points (n-1/2)*f. Octave
    errors are suppressed because a sub-octave candidate hits half of its
    in-between points on real harmonics, and the upper octave hits real
    harmonics at its in-between points. Multiples k*f0 (k = 2, 3, 4) are
    resolved by the subharmonic preference below.
    """
    t, f, X = spec if spec is not None else stft(x, fs, hop=hop)
    S = 20 * np.log10(X + 1e-9)
    cents = np.arange(0, 1200 * np.log2(fmax / fmin), 5.0)
    cand = fmin * 2 ** (cents / 1200)
    n = np.arange(1, n_harm + 1)
    scores = np.empty((len(t), len(cand)))
    for j, fc in enumerate(cand):
        on = _sample(S, f, np.broadcast_to(n * fc, (len(t), n_harm)))
        off = _sample(S, f, np.broadcast_to((n - 0.5) * fc, (len(t), n_harm)))
        scores[:, j] = on.mean(1) - off.mean(1)
    j = np.argmax(scores, axis=1)
    # subharmonic preference: a multiple k*f of the true f0 also scores high
    # when the in-between points of k*f are weak; prefer f/k if its score is
    # within ``sub_margin`` dB of the maximum
    rows = np.arange(len(t))
    for k in (4, 3, 2):
        js = j - int(round(1200 * np.log2(k) / 5.0))
        valid = js >= 0
        jj_ = np.clip(js, 0, len(cand) - 1)
        better = valid & (scores[rows, jj_] >= scores[rows, j] - sub_margin)
        j = np.where(better, jj_, j)
    best_s = scores[rows, j]
    # parabolic refinement on the cent grid
    jj = np.clip(j, 1, len(cand) - 2)
    y0, y1, y2 = scores[rows, jj - 1], scores[rows, jj], scores[rows, jj + 1]
    den = y0 - 2 * y1 + y2
    delta = np.where(np.abs(den) > 1e-9, 0.5 * (y0 - y2) / den, 0.0)
    best_f = fmin * 2 ** ((cents[jj] + np.clip(delta, -1, 1) * 5.0) / 1200)

    rms = 20 * np.log10(np.sqrt((X ** 2).sum(1) / 2) + 1e-9)
    f0 = np.where((best_s > min_score) & (rms > rms.max() - 45), best_f, np.nan)
    return F0Track(t, f0, best_s, rms)


def harmonic_amplitudes(spec, f0, n_harm=60, fmax=8000.0):
    """dB amplitude of harmonics n*f0 for every frame (NaN above fmax or unvoiced).

    The peak is searched in a +-3 % window around n*f0, which tolerates small
    f0 errors at high harmonic numbers.
    """
    t, f, X = spec
    S = 20 * np.log10(X + 1e-9)
    n = np.arange(1, n_harm + 1)
    out = np.full((len(t), n_harm), np.nan)
    ok = np.isfinite(f0)
    if not ok.any():
        return out
    hf = f0[ok, None] * n[None, :]
    best = np.full(hf.shape, -np.inf)
    for r in np.linspace(-0.03, 0.03, 13):
        q = np.clip(hf * (1 + r / np.sqrt(n)), 0, f[-2])
        best = np.maximum(best, _sample(S[ok], f, q))
    best[hf > fmax] = np.nan
    out[ok] = best
    return out
