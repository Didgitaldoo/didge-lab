"""
Frame analysis of didgeridoo recordings (analysis-by-synthesis).

For every 10 ms frame of a recording ``analyse`` computes

1. the lip frequency, constrained to the DidgeLab resonances of the
   instrument, and the resonance it is played on (drone / toot k)
   (``track_lip_f0``);
2. the tongue position, by matching the spectral envelope with the
   vocal-tract model along the tongue axis (``tongue_track``), and the model
   formant of that position;
3. a grid inversion of the full vocal tract with Viterbi smoothing
   (``VTInverter``; ill-posed, used only as additional classifier features);
4. features: level, inharmonic energy, non-harmonic high-band noise,
   20-40 Hz amplitude modulation, band energies.

The techniques are detected from these frames by the learned detector in
``learn.py`` (see ``transcribe.py``). ``detect`` below is a simple rule-based
fallback used when no trained model is available:

* toots                f0 on the 2nd (3rd ...) resonance
* stop consonants      short dips of the level; t/d vs. k/g from the tongue
* trills (rolled r)    20-40 Hz amplitude modulation
* vocalisation         energy between the harmonics
* circular breathing   sealed-mouth configuration and nasal noise

Ported from didge-synth (``didgesynth.detect``).
"""

import itertools
from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import median_filter, uniform_filter1d
from scipy.signal import find_peaks

from .vocaltract import ARTICULATED_MOUTH

from . import analysis
from .model import ModelParams, filter_gain, source

HOP = 480          # 10 ms at 48 kHz
FS = 48000


@dataclass
class FrameAnalysis:
    t: np.ndarray
    f0: np.ndarray
    level_db: np.ndarray
    mode: np.ndarray            # 0 = drone, k = k-th toot, -1 = unvoiced
    vt: np.ndarray              # (T, 4) fitted vocal tract
    fit_err: np.ndarray         # dB rms residual of the inversion
    inharm: np.ndarray          # inharmonic energy ratio
    hf_noise_db: np.ndarray     # level of non-harmonic energy 2-6 kHz rel. to tone
    harm_db: np.ndarray         # (T, n_harm) measured harmonics
    am_rate: np.ndarray         # dominant amplitude modulation rate 15-45 Hz
    am_depth: np.ndarray        # its depth (dB)
    bands_db: np.ndarray = None # (T, 9) band energies relative to the frame level
    formant_hz: np.ndarray = None   # model formant of the matched tongue position, Hz
    tongue_match: np.ndarray = None # correlation of the envelope with the matched template
    tongue_index: np.ndarray = None # tongue position (0 front .. 1 back)


@dataclass
class Segment:
    start: float
    end: float
    label: str
    info: dict = field(default_factory=dict)


class VTInverter:
    """Grid search inversion of the vocal tract from harmonic amplitudes."""

    def __init__(self, bore, params=ModelParams(), n_harm=50, fmax=4500.0,
                 grid=None):
        self.bore = bore
        self.params = params
        self.n_harm = n_harm
        self.fmax = fmax
        if grid is None:
            grid = np.array(list(itertools.product(np.linspace(0, 1, 11), np.linspace(0, 0.95, 8),
                                                   [0.1, 0.5, 0.9], [0.05, 0.3, 0.7])))
        self.grid = grid
        self._cache = {}

    def table(self, f0):
        key = int(round(1200 * np.log2(f0 / 50.0) / 10))      # 10 cent bins
        if key not in self._cache:
            fq = 50.0 * 2 ** (key * 10 / 1200)
            f = fq * np.arange(1, self.n_harm + 1)
            F = np.tile(f, (len(self.grid), 1))
            src = source(self.params, self.n_harm)
            G = filter_gain(self.bore, F, self.grid, self.params)
            M = 20 * np.log10(np.abs(G * src[None, :]) + 1e-30)
            self._cache[key] = (f, M)
        return self._cache[key]

    def errors(self, harm_db, f0):
        """RMS error (dB) of every grid configuration for one frame, after
        fitting gain and spectral tilt; None if too few harmonics."""
        f, M = self.table(f0)
        m = np.isfinite(harm_db) & (f <= self.fmax)
        if m.sum() < 6:
            return None
        X = np.stack([np.ones(m.sum()), np.log2(f[m] / 1000)], 1)
        P = X @ np.linalg.pinv(X)
        D = harm_db[m][None, :] - M[:, m]
        res = D - D @ P.T
        return np.sqrt(np.mean(res ** 2, axis=1))

    def fit(self, harm_db, f0):
        """Return (grid index, rms error) for one frame."""
        e = self.errors(harm_db, f0)
        if e is None:
            return -1, np.nan
        j = int(np.argmin(e))
        return j, float(e[j])

    def track(self, H, f0, weights=(4.0, 3.0, 1.5, 1.5)):
        """Viterbi path through the grid over all frames.

        Cost = fit error (dB) + sum_k w_k |delta p_k| between consecutive
        frames: articulators move continuously, so jumps across the grid
        must be justified by a clearly better fit.
        """
        T = len(f0)
        C = len(self.grid)
        trans = (np.abs(self.grid[:, None, :] - self.grid[None, :, :]) * np.asarray(weights)).sum(-1)
        idx = np.full(T, -1)
        err = np.full(T, np.nan)
        frames = [i for i in range(T) if np.isfinite(f0[i])]
        E = {}
        for i in frames:
            e = self.errors(H[i], f0[i])
            if e is not None:
                E[i] = e
        # run Viterbi on consecutive stretches of fitted frames
        keys = sorted(E)
        stretches, cur = [], []
        for i in keys:
            if cur and i != cur[-1] + 1:
                stretches.append(cur)
                cur = []
            cur.append(i)
        if cur:
            stretches.append(cur)
        for st in stretches:
            cost = E[st[0]].copy()
            back = np.zeros((len(st), C), dtype=np.int32)
            for n, i in enumerate(st[1:], 1):
                tot = cost[:, None] + trans
                back[n] = np.argmin(tot, axis=0)
                cost = tot[back[n], np.arange(C)] + E[i]
            j = int(np.argmin(cost))
            for n in range(len(st) - 1, -1, -1):
                idx[st[n]] = j
                err[st[n]] = E[st[n]][j]
                j = back[n, j]
        return idx, err


def analyse(x, fs, bore, params=ModelParams(), n_harm=50, invert=True):
    """Frame-wise analysis of a recording (see module docstring)."""
    if fs != FS:
        raise ValueError("resample to 48 kHz first (analysis.load_audio(path, 48000))")
    spec = analysis.stft(x, fs, hop=HOP)
    res = np.array([r.freq for r in bore.resonances[:6]])
    tr = analysis.track_f0(x, fs, fmin=0.8 * res[0], fmax=1.15 * res[-1], spec=spec)
    f0, mode = track_lip_f0(spec, res, voiced=np.isfinite(tr.f0) | (tr.rms_db > np.nanmax(tr.rms_db) - 30))
    t = tr.t
    T = len(t)
    ok = np.isfinite(f0)

    H = analysis.harmonic_amplitudes(spec, f0, n_harm=n_harm, fmax=6000)
    inh = analysis.inharmonic_ratio(spec, f0, fmin=60, fmax=2500)

    # non-harmonic high band energy (nasal sniff, frication)
    _, f, X = spec
    P = X ** 2
    hb = (f > 2000) & (f < 6000)
    hf = np.full(T, np.nan)
    for i in np.where(ok)[0]:
        r = f[hb] / f0[i]
        between = np.abs(r - np.round(r)) > 0.25
        hf[i] = 10 * np.log10(P[i, hb][between].mean() + 1e-20)
    level = tr.rms_db
    hf_rel = hf - level

    # amplitude modulation of the 10 ms level envelope (trills)
    am_rate, am_depth = _am_track(x, fs, T)

    # band energies relative to the frame level (timbre without the vocal-tract model)
    edges = [60, 200, 400, 700, 1100, 1600, 2300, 3200, 4500, 7000]
    bands = np.stack([10 * np.log10(P[:, (f >= lo) & (f < hi)].sum(1) + 1e-20)
                      for lo, hi in zip(edges[:-1], edges[1:])], 1)
    bands = bands - 10 * np.log10(P[:, (f >= 60) & (f < 7000)].sum(1) + 1e-20)[:, None]

    tongue, match = tongue_track(spec, ok, bore, params)
    F = formant_from_tongue(tongue, bore, params)

    vt = np.full((T, 4), np.nan)
    err = np.full(T, np.nan)
    if invert:
        inv = VTInverter(bore, params, n_harm=n_harm)
        idx, err = inv.track(H, f0)
        vt[idx >= 0] = inv.grid[idx[idx >= 0]]
    return FrameAnalysis(t, f0, level, mode, vt, err, inh, hf_rel, H, am_rate, am_depth, bands,
                         F, match, tongue)


_TONGUE_CURVES = {}


def tongue_curve(bore, params=ModelParams(), height=0.5, mouth=ARTICULATED_MOUTH, glottis=0.3):
    """Output formant of the model as a function of tongue position.

    Along the tongue-position axis (0.4 .. 0.95, where the formant falls
    monotonically) the model formant is computed with the same envelope
    envelope method as ``tongue_track``. Returns (tongue_pos, formant_hz).
    """
    key = (id(bore), height, mouth, glottis)
    if key in _TONGUE_CURVES:
        return _TONGUE_CURVES[key]
    from scipy.ndimage import uniform_filter1d
    ff = np.arange(300.0, 4000.0, 5.0)
    ref = 20 * np.log10(np.abs(filter_gain(bore, ff, None, params)) + 1e-30)
    lf = np.log2(ff)
    A = np.stack([np.ones_like(lf), lf], 1)
    P = A @ np.linalg.pinv(A)
    tps = np.linspace(0.4, 0.95, 23)
    search = ff >= 600.0
    F = []
    for tp in tps:
        g = 20 * np.log10(np.abs(filter_gain(bore, ff, np.array([tp, height, mouth, glottis]), params)) + 1e-30)
        D = uniform_filter1d(g - ref, 40)
        R = D - P @ D
        # above the first mouth resonance (~500 Hz with an open jaw), which would
        # otherwise win at the front positions and flatten the whole curve
        F.append(ff[search][np.argmax(R[search])])
    F = np.minimum.accumulate(np.array(F))     # enforce monotonic fall
    _TONGUE_CURVES[key] = (tps, F)
    return tps, F


TONGUE_GRID = np.linspace(0.0, 1.0, 41)
TONGUE_FMIN, TONGUE_FMAX = 600.0, 3000.0
TONGUE_MIN_MATCH = 0.3
_TONGUE_TEMPLATES = {}


def tongue_vt(x, height=0.5, mouth=ARTICULATED_MOUTH, glottis=0.3):
    """Articulated mouth along the tongue axis: 0 = front ("ee") .. 1 = back ("oo")."""
    return np.array([0.4 + 0.55 * x, height, mouth, glottis])


def _envelope(S_db, f, df):
    """Spectral envelope: maximum over +-40 Hz (bridges the harmonics), smoothed over 250 Hz."""
    from scipy.ndimage import maximum_filter1d
    E = maximum_filter1d(S_db, int(80 / df) | 1, axis=-1)
    return uniform_filter1d(E, int(250 / df) | 1, axis=-1)


def tongue_templates(bore, freqs, params=ModelParams()):
    """Change of the radiated spectrum (dB) for every position of ``TONGUE_GRID``,
    relative to the mean over all positions; shape (len(TONGUE_GRID), len(freqs))."""
    key = (id(bore), len(freqs), float(freqs[0]), float(freqs[-1]))
    if key in _TONGUE_TEMPLATES:
        return _TONGUE_TEMPLATES[key]
    ref = 20 * np.log10(np.abs(filter_gain(bore, freqs, None, params)) + 1e-30)
    D = np.array([20 * np.log10(np.abs(filter_gain(bore, freqs, tongue_vt(x), params)) + 1e-30) - ref
                  for x in TONGUE_GRID])
    D = uniform_filter1d(D, max(int(250 / (freqs[1] - freqs[0])), 1) | 1, axis=1)
    D = D - D.mean(0)
    _TONGUE_TEMPLATES[key] = D
    return D


def tongue_track(spec, voiced, bore, params=ModelParams(), min_match=TONGUE_MIN_MATCH):
    """Tongue position per frame (0 front .. 1 back) by template matching.

    The spectral envelope of each frame minus the median envelope of the voiced
    frames (this removes the bore, the radiation and the recording chain) is
    correlated, between 600 Hz and 3 kHz, with the change of the model spectrum
    along the tongue axis (``tongue_templates``). The best template, refined by
    a parabola through the correlations, gives the tongue position. Returns
    (tongue_index, match); the index is NaN where the correlation is below
    ``min_match`` or the frame is unvoiced.

    Because the reference is the recording's own median, the index measures the
    tongue relative to the player's average position: movements are measured
    reliably, absolute front/back shares are not.
    """
    t, f, X = spec
    m = (f >= TONGUE_FMIN) & (f <= TONGUE_FMAX)
    fm = f[m]
    E = _envelope(20 * np.log10(X[:, m] + 1e-9), fm, f[1] - f[0])
    ok = np.asarray(voiced, bool)
    if not ok.any():
        return np.full(len(t), np.nan), np.full(len(t), np.nan)
    R = E - np.median(E[ok], axis=0)
    T = tongue_templates(bore, fm, params)
    Rc = R - R.mean(1, keepdims=True)
    Tc = T - T.mean(1, keepdims=True)
    C = (Rc @ Tc.T) / (np.linalg.norm(Rc, axis=1)[:, None] * np.linalg.norm(Tc, axis=1)[None] + 1e-9)
    j = np.argmax(C, axis=1)
    rows = np.arange(len(C))
    jj = np.clip(j, 1, len(TONGUE_GRID) - 2)
    a, b, c = C[rows, jj - 1], C[rows, jj], C[rows, jj + 1]
    den = a - 2 * b + c
    shift = np.where((j == jj) & (den < 0), 0.5 * (a - c) / np.where(den < 0, den, -1), 0.0)
    step = TONGUE_GRID[1] - TONGUE_GRID[0]
    x = np.clip(TONGUE_GRID[j] + np.clip(shift, -0.5, 0.5) * step, 0.0, 1.0)
    match = C[rows, j]
    good = ok & (match >= min_match)
    return np.where(good, x, np.nan), np.where(ok, match, np.nan)


def formant_from_tongue(x, bore, params=ModelParams()):
    """Model formant (Hz) at tongue index ``x`` (inverse of ``tongue_from_formant``)."""
    tps, Fc = tongue_curve(bore, params)
    return np.interp(0.4 + 0.55 * np.asarray(x, float), tps, Fc, left=np.nan, right=np.nan)


def tongue_from_formant(F, bore, params=ModelParams()):
    """Tongue index (0 = front, 1 = back) from the measured formant."""
    tps, Fc = tongue_curve(bore, params)
    # rescale the model's tongue range 0.4..0.95 to 0..1 and extrapolate
    # linearly in log-frequency beyond the model's formant range
    x = np.log2(Fc[::-1])
    y = ((tps - 0.4) / 0.55)[::-1]
    lf = np.log2(np.where(np.isfinite(F), F, np.nan))
    slope = (y[-1] - y[0]) / (x[-1] - x[0]) if x[-1] != x[0] else -1.0
    out = np.interp(lf, x, y)
    out = np.where(lf < x[0], y[0] + slope * (lf - x[0]), out)
    out = np.where(lf > x[-1], y[-1] + slope * (lf - x[-1]), out)
    return np.clip(out, -0.25, 1.25)


def track_lip_f0(spec, resonances, voiced, window_cents=90.0, margin_db=4.0, n_harm=12):
    """Lip frequency constrained to the bore resonances.

    The lips of a didgeridoo player lock to an impedance peak, so candidates
    are searched only within +-``window_cents`` of each resonance. The score
    of a candidate is the mean level at its first ``n_harm`` harmonics
    relative to the mean level of the frame. The lowest resonance whose score
    is within ``margin_db`` of the best one wins: a toot on resonance k is
    only accepted when the drone's harmonics in between are clearly weaker.
    This is robust against vocalisation sidebands, which fall between the
    harmonics and would break a harmonic-contrast score.
    """
    t, f, X = spec
    S = 20 * np.log10(X + 1e-9)
    band = (f > 30) & (f < 3000)
    mean_level = S[:, band].mean(1)
    n = np.arange(1, n_harm + 1)
    best_f = np.full((len(t), len(resonances)), np.nan)
    best_s = np.full((len(t), len(resonances)), -np.inf)
    for k, fr in enumerate(resonances):
        for c in np.arange(-window_cents, window_cents + 1, 5.0):
            fc = fr * 2 ** (c / 1200)
            q = np.broadcast_to(n * fc, (len(t), n_harm))
            sc = analysis._sample(S, f, q).mean(1) - mean_level
            better = sc > best_s[:, k]
            best_s[better, k] = sc[better]
            best_f[better, k] = fc
    top = best_s.max(1)
    mode = np.argmax(best_s >= top[:, None] - margin_db, axis=1)
    f0 = best_f[np.arange(len(t)), mode]
    f0 = np.where(voiced, f0, np.nan)
    mode = np.where(voiced, mode, -1)
    return f0, mode


def _am_track(x, fs, T, hop=HOP):
    """Dominant 15-45 Hz modulation of the signal envelope in 200 ms windows."""
    env_rate = 400
    k = fs // env_rate
    e = np.sqrt(uniform_filter1d(x ** 2, k)[::k] + 1e-12)
    e_db = 20 * np.log10(e)
    win = int(0.2 * env_rate)
    rate = np.full(T, np.nan)
    depth = np.full(T, 0.0)
    freqs = np.fft.rfftfreq(win * 4, 1 / env_rate)
    band = (freqs >= 15) & (freqs <= 45)
    w = np.hanning(win)
    for i in range(T):
        c = int(i * hop / fs * env_rate)
        s = c - win // 2
        if s < 0 or s + win > len(e_db):
            continue
        seg = e_db[s:s + win]
        seg = seg - np.polyval(np.polyfit(np.arange(win), seg, 1), np.arange(win))
        S = np.abs(np.fft.rfft(seg * w, win * 4)) * 2 / w.sum()
        j = np.argmax(S[band])
        rate[i] = freqs[band][j]
        depth[i] = S[band][j]
    return rate, depth


# ---------------------------------------------------------------------------
# event detection
def _runs(mask, t, min_len=0.0):
    out = []
    i = 0
    n = len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            if t[j - 1] - t[i] + (t[1] - t[0]) >= min_len:
                out.append((i, j))
            i = j
        else:
            i += 1
    return out


@dataclass
class DetectorConfig:
    dip_db: float = 4.0            # minimum level dip of a stop consonant
    dip_deep_db: float = 9.0       # dips deeper than this are unvoiced (t/k)
    dip_max_len: float = 0.09      # s
    front_back_split: float = 0.45 # tongue position separating t/d from k/g
    trill_depth_db: float = 2.5
    voice_inharm: float = 0.22
    breath_seal_tp: float = 0.6    # fitted tongue position of a sealed mouth
    breath_seal_th: float = 0.75
    breath_hf_db: float = 6.0      # sniff noise rise above the local median
    breath_min_len: float = 0.2    # s; consonants also raise the high-band noise, but briefly
                                   # (synthetic performances: F1 0.42 at 0.12 s, 0.72 at 0.2 s)
    toot_min_len: float = 0.08


def detect(fa: FrameAnalysis, cfg=DetectorConfig()):
    """Detect technique events from a FrameAnalysis; returns list of Segment."""
    t = fa.t
    dt = t[1] - t[0]
    segs = []
    voiced = fa.mode >= 0

    # --- toots
    for i, j in _runs(fa.mode >= 1, t, cfg.toot_min_len):
        k = int(np.median(fa.mode[i:j]))
        segs.append(Segment(t[i], t[j - 1] + dt, f"toot{k}", {"f0": float(np.nanmedian(fa.f0[i:j]))}))

    # --- stop consonants: dips of the level relative to a running median
    lev = np.where(np.isfinite(fa.level_db), fa.level_db, np.nanmin(fa.level_db))
    ref = median_filter(lev, size=int(0.25 / dt) | 1, mode="nearest")
    dip = ref - lev
    peaks, props = find_peaks(dip, height=cfg.dip_db, distance=int(0.06 / dt),
                              width=(1, cfg.dip_max_len / dt))
    loud = lev > np.nanpercentile(lev, 20)
    for p, h in zip(peaks, props["peak_heights"]):
        if not loud[max(p - 10, 0)]:
            continue
        lo, hi = max(0, p - int(0.05 / dt)), min(len(t), p + int(0.05 / dt))
        tp = np.nanmedian(fa.vt[lo:hi, 0]) if np.any(np.isfinite(fa.vt[lo:hi, 0])) else np.nan
        front = (not np.isfinite(tp)) or tp < cfg.front_back_split
        deep = h >= cfg.dip_deep_db
        consonant = ("t" if front else "k") if deep else ("d" if front else "g")
        # same labels as the learned detector (learn.CLASSES)
        segs.append(Segment(t[p] - 0.02, t[p] + 0.02, "stop_front" if front else "stop_back",
                            {"consonant": consonant, "voiced": not deep, "depth_db": float(h),
                             "tongue_pos": float(tp) if np.isfinite(tp) else None}))

    # --- trills
    trill = (fa.am_depth > cfg.trill_depth_db) & voiced
    trill = median_filter(trill.astype(int), size=7) > 0
    for i, j in _runs(trill, t, 0.12):
        segs.append(Segment(t[i], t[j - 1] + dt, "trill", {"rate_hz": float(np.nanmedian(fa.am_rate[i:j]))}))

    # --- vocalisation
    inh = np.where(np.isfinite(fa.inharm), fa.inharm, 0)
    inh_s = median_filter(inh, size=9)
    for i, j in _runs((inh_s > cfg.voice_inharm) & (fa.level_db > np.nanpercentile(fa.level_db, 25)), t, 0.15):
        segs.append(Segment(t[i], t[j - 1] + dt, "voice", {"inharm": float(np.mean(inh_s[i:j]))}))

    # --- circular breathing: sealed-mouth configuration + nasal noise
    vt = fa.vt
    sealed = (vt[:, 0] >= cfg.breath_seal_tp) & (vt[:, 1] >= cfg.breath_seal_th) & (vt[:, 2] <= 0.3)
    hf = np.where(np.isfinite(fa.hf_noise_db), fa.hf_noise_db, np.nan)
    hf_ref = median_filter(np.nan_to_num(hf, nan=np.nanmedian(hf)), size=int(1.0 / dt) | 1)
    sniff = np.nan_to_num(hf - hf_ref, nan=0) > cfg.breath_hf_db
    sniff = uniform_filter1d(sniff.astype(float), int(0.1 / dt)) > 0.3
    cand = median_filter((sealed | sniff).astype(int), size=5) > 0
    for i, j in _runs(cand, t, cfg.breath_min_len):
        if t[j - 1] - t[i] > 0.8:
            continue
        score = float(np.mean(sealed[i:j]) + np.mean(sniff[i:j]))
        if score > 0.5:
            segs.append(Segment(t[i], t[j - 1] + dt, "circular_breath", {"score": score}))

    segs.sort(key=lambda s: s.start)
    return segs


def tongue_summary(fa: FrameAnalysis, split=(0.35, 0.65)):
    """Share of voiced time with the tongue front / middle / back.

    The tongue index is relative to the player's average position (see
    ``tongue_track``), so these shares describe the spread of the movement,
    not the absolute position."""
    tp = fa.tongue_index if fa.tongue_index is not None else fa.vt[:, 0]
    tp = tp[np.isfinite(tp)]
    if len(tp) == 0:
        return {}
    tp = median_filter(tp, size=5)
    return {"front": float(np.mean(tp < split[0])), "middle": float(np.mean((tp >= split[0]) & (tp <= split[1]))),
            "back": float(np.mean(tp > split[1]))}


def summarize(fa: FrameAnalysis, segs):
    dur = fa.t[-1] - fa.t[0]
    counts = {}
    for s in segs:
        counts[s.label] = counts.get(s.label, 0) + 1
    voiced = np.mean(fa.mode >= 0)
    drone = np.nanmedian(fa.f0[fa.mode == 0]) if np.any(fa.mode == 0) else np.nan
    return {
        "duration_s": float(dur),
        "voiced_fraction": float(voiced),
        "drone_hz": float(drone),
        "events": counts,
        "events_per_s": {k: v / dur for k, v in counts.items()},
        "tongue": tongue_summary(fa),
        "median_fit_err_db": float(np.nanmedian(fa.fit_err)),
    }
