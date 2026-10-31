"""
Signal analysis of didgeridoo recordings.

* robust f0 tracking (harmonic-contrast score, refined by parabolic interpolation)
* harmonic amplitudes per frame
* frame features (RMS, spectral centroid, inharmonic energy, noise)

Ported from didge-synth (``didgesynth.analysis``).
"""

from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import resample_poly


def load_audio(path, fs=None, mono=True):
    """Read an audio file as float mono, optionally resampled to ``fs``.

    Uses ``soundfile`` if installed (wav, flac, ogg, ...), else scipy (wav only).
    """
    try:
        import soundfile as sf
        x, sr = sf.read(path, always_2d=True)
    except ImportError:
        from scipy.io import wavfile
        sr, x = wavfile.read(path)
        if np.issubdtype(x.dtype, np.integer):
            x = x / float(np.iinfo(x.dtype).max)
        x = np.asarray(x, dtype=np.float64).reshape(len(x), -1)
    x = x.mean(axis=1) if mono else x
    if fs is not None and sr != fs:
        g = np.gcd(int(sr), int(fs))
        x = resample_poly(x, fs // g, sr // g, axis=0)
        sr = fs
    return x.astype(np.float64), int(sr)


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


def inharmonic_ratio(spec, f0, fmin=60.0, fmax=3000.0):
    """Fraction of spectral energy that is *not* within +-1/6 f0 of a harmonic.

    High values indicate vocalisation (sum/difference tones), noise or
    unpitched sounds.
    """
    t, f, X = spec
    P = X ** 2
    band = (f >= fmin) & (f <= fmax)
    out = np.full(len(t), np.nan)
    for i in np.where(np.isfinite(f0))[0]:
        ff = f[band]
        r = ff / f0[i]
        harm = np.abs(r - np.round(r)) < 1 / 6
        tot = P[i, band].sum() + 1e-20
        out[i] = 1 - P[i, band][harm].sum() / tot
    return out


def spectral_centroid(spec, fmax=8000.0):
    t, f, X = spec
    m = f <= fmax
    P = X[:, m] ** 2
    return (P * f[m]).sum(1) / (P.sum(1) + 1e-20)


def ltas_harmonics(H):
    """Long-term mean harmonic profile in dB (power average over frames)."""
    P = 10 ** (H / 10)
    with np.errstate(invalid="ignore"):
        m = np.nanmean(P, axis=0)
    return 10 * np.log10(m + 1e-20)


def smooth_f0(f0, size=9):
    g = f0.copy()
    ok = np.isfinite(g)
    if ok.sum() > size:
        g[ok] = median_filter(g[ok], size=size, mode="nearest")
    return g


def formant_track(spec, fmin=500.0, fmax=4000.0, voiced=None, min_prom=4.0, prefer_db=4.0):
    """Dominant formant of the didgeridoo sound, model-free.

    The spectral envelope of each frame (maximum over +-40 Hz to bridge the
    harmonics, smoothed over 250 Hz) is compared with the long-term envelope
    of the recording; level and tilt are removed by a linear fit over
    log-frequency. The frequency of the largest remaining peak is the
    formant, its height above the median the prominence (dB).
    Returns (formant_hz, prominence_db); formant is NaN where prominence < min_prom.
    """
    from scipy.ndimage import maximum_filter1d, uniform_filter1d
    t, f, X = spec
    S = 20 * np.log10(X + 1e-9)
    m = (f >= fmin) & (f <= fmax)
    fm = f[m]
    df = f[1] - f[0]
    E = maximum_filter1d(S[:, m], int(80 / df) | 1, axis=1)
    E = uniform_filter1d(E, int(250 / df) | 1, axis=1)
    sel = voiced if voiced is not None and np.any(voiced) else slice(None)
    ref = np.median(E[sel], axis=0)
    R = E - ref
    lf = np.log2(fm)
    A = np.stack([np.ones_like(lf), lf], 1)
    R = R - R @ (A @ np.linalg.pinv(A)).T
    # the lowest envelope peak within ``prefer_db`` of the strongest one: the
    # second formant (bright line above the first) is often almost as strong
    from scipy.signal import find_peaks
    k = np.argmax(R, axis=1)
    for i in range(len(R)):
        pk, _ = find_peaks(R[i])
        pk = pk[R[i, pk] >= R[i, k[i]] - prefer_db]
        if len(pk):
            k[i] = pk[0]
    prom = R[np.arange(len(R)), k] - np.median(R, axis=1)
    F = fm[k]
    F = np.where(prom >= min_prom, F, np.nan)
    if voiced is not None:
        F = np.where(voiced, F, np.nan)
    return F, prom




def sideband_level(x, fs, f0, fmax=1500.0, min_frac=0.2):
    """Level (dB re. the strongest partial) of the strongest spectral peak
    that is not a harmonic of f0 (|f/f0 - round(f/f0)| > min_frac).

    Singing into the didgeridoo creates strong sum and difference tones
    n*f0 +- k*fv between the harmonics; without vocalisation such peaks are
    25-35 dB below the strongest partial.
    """
    from scipy.signal import find_peaks
    N = len(x)
    if N < fs // 10 or not np.isfinite(f0):
        return -np.inf
    X = 20 * np.log10(np.abs(np.fft.rfft(x * np.hanning(N), 4 * N)) + 1e-12)
    f = np.fft.rfftfreq(4 * N, 1 / fs)
    m = (f > 40) & (f < fmax)
    X, f = X[m] - X[m].max(), f[m]
    p, _ = find_peaks(X, prominence=8)
    r = f[p] / f0
    inh = np.abs(r - np.round(r)) > min_frac
    return float(X[p][inh].max()) if inh.any() else -np.inf
