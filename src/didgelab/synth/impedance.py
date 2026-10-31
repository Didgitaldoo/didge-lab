"""
Estimating the input impedance of an unknown didgeridoo from a sound sample.

Nothing about the bore is known. The microphone hears a transfer function,
not Z_in, so the bore is treated as the unknown of the forward model of
``didgelab.synth.model`` and fitted to what the sample reveals:

* the drone pitch f0 -- the first impedance maximum;
* the harmonic levels -- |G(n f0)|, sampled every f0. Formants only ever
  boost harmonics, so a low percentile over frames approximates the
  neutral mouth (no technique detection needed);
* free decays -- after a note ends or a slap, the bore rings at its own
  resonances; a matrix-pencil fit gives their frequencies and Q directly.

The bore is a piecewise conical geometry with a few free diameters. Many
geometries share an impedance curve, so the geometry itself is not the
result; its Z_in is. Restarts of the global optimiser give an uncertainty
band. The absolute scale of |Z| is set by the assumed mouthpiece diameter.

Ported from didge-synth (``didgesynth.impedance``); the method and its
evaluation on the Didge Archive are described in doc/impedance_from_sound.md.

Example::

    from didgelab.synth import estimate_impedance, render_song_loop_from_peaks
    est = estimate_impedance(x, fs)          # mono float array, any sample rate >= 8 kHz
    est.resonances                           # [(freq_hz, |Z| MPa s/m^3, Q), ...]
    y, fs = render_song_loop_from_peaks(est.peaks[:6], "tiki-du")
"""

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import differential_evolution
from scipy.signal import find_peaks, butter, sosfiltfilt, resample_poly

from .analysis import stft, track_f0, harmonic_amplitudes
from .bore import Bore, Resonance, chain_matrix
from .model import ModelParams, source
from .vocaltract import VOWELS, VocalTract, impedance as vt_impedance
from ..geo import Geo
from ..sim.sim_interface import compute_moist_air_properties

MOUTHPIECE_MM = 31.0
KNOTS = np.array([0.0, 0.15, 0.35, 0.55, 0.75, 0.9, 1.0])   # relative positions of the free diameters
N_SEG = 24
PEAK_GRID = 30.0 * 2 ** (np.arange(0, 1200 * np.log2(1600 / 30.0), 4.0) / 1200)


# ---------------------------------------------------------------------------
# observations
@dataclass
class DecayMode:
    t: float          # time of the decay (s)
    freq: float       # Hz
    q: float
    level_db: float   # relative to the strongest mode of that decay


@dataclass
class Observations:
    f0: float                                   # median drone pitch, NaN if no drone
    harm_f: np.ndarray = field(default_factory=lambda: np.zeros(0))    # Hz
    harm_db: np.ndarray = field(default_factory=lambda: np.zeros(0))   # low-percentile level, dB
    decays: list = field(default_factory=list)  # DecayMode
    voiced_s: float = 0.0

    def mode_clusters(self, width_cents=40.0, min_count=1):
        """Decay modes grouped by frequency: (freq, count). A resonance rings
        in many decays, a room mode or a fitting artefact in few."""
        f = np.sort([d.freq for d in self.decays])
        out = []
        for v in f:
            if out and 1200 * np.log2(v / out[-1][-1]) < width_cents:
                out[-1].append(v)
            else:
                out.append([v])
        return [(float(np.median(c)), len(c)) for c in out if len(c) >= min_count]


def _rms_db(x, fs, hop):
    n = len(x) // hop
    fr = x[:n * hop].reshape(n, hop)
    return 10 * np.log10(np.mean(fr ** 2, axis=1) + 1e-14)


def matrix_pencil(y, fs, order=16):
    """Damped complex exponentials of a real signal: (freq Hz, damping 1/s, amplitude)."""
    N = len(y)
    L = N // 3
    Y = np.lib.stride_tricks.sliding_window_view(y, L + 1)
    U, s, Vh = np.linalg.svd(Y, full_matrices=False)
    M = min(order, int(np.sum(s > s[0] * 10 ** (-50 / 20))))
    if M < 2:
        return np.zeros((0, 3))
    V = Vh[:M].conj().T
    z = np.linalg.eigvals(np.linalg.pinv(V[:-1]) @ V[1:])
    Z = z[None, :] ** np.arange(N)[:, None]
    amp = np.abs(np.linalg.lstsq(Z, y.astype(complex), rcond=None)[0])
    f = np.angle(z) * fs / (2 * np.pi)
    a = -np.log(np.abs(z) + 1e-30) * fs
    keep = f > 0
    return np.stack([f[keep], a[keep], 2 * amp[keep]], 1)


def find_decays(x, fs, hop_s=0.005, drop_db=15.0, within_s=0.15, length_s=0.12,
                fmin=40.0, fmax=1200.0):
    """Free decays: a level maximum followed by a fall of ``drop_db`` within
    ``within_s``. Each decay is fitted with a matrix pencil (low-passed and
    decimated to 4 kHz); modes with a plausible Q are returned."""
    hop = int(hop_s * fs)
    r = _rms_db(x, fs, hop)
    k = int(within_s / hop_s)
    top = r.max()
    cand = []
    for i in range(len(r) - k):
        if r[i] < top - 35 or r[i] < r[max(0, i - 4):i + 2].max():
            continue
        if r[i] - r[i + 1:i + k + 1].min() >= drop_db:
            cand.append(i)
    # one decay per 250 ms
    picked = []
    for i in cand:
        if not picked or (i - picked[-1]) * hop_s > 0.25:
            picked.append(i)
    sos = butter(6, fmax * 1.3, fs=fs, output="sos")
    xl = sosfiltfilt(sos, x)
    fsd = 4000
    g = np.gcd(int(fs), fsd)
    out = []
    for i in picked:
        s0 = (i + 2) * hop                      # skip the excitation itself
        seg = xl[s0:s0 + int(length_s * fs)]
        if len(seg) < 0.05 * fs:
            continue
        seg = resample_poly(seg, fsd // g, int(fs) // g)
        modes = matrix_pencil(seg - seg.mean(), fsd)
        if not len(modes):
            continue
        f, a, amp = modes.T
        q = np.pi * f / np.maximum(a, 1e-3)
        lvl = 20 * np.log10(amp / amp.max() + 1e-12)
        ok = (f > fmin) & (f < fmax) & (q > 3) & (q < 80) & (lvl > -30)
        out += [DecayMode(i * hop_s, float(ff), float(qq), float(ll))
                for ff, qq, ll in zip(f[ok], q[ok], lvl[ok])]
    return out


def observe(x, fs, fmax=3000.0, percentile=25.0, min_voiced_s=0.3):
    """Extract drone pitch, low-percentile harmonic levels and free decays."""
    spec = stft(x, fs)
    tr = track_f0(x, fs, fmin=40.0, fmax=200.0, spec=spec)
    ok = np.isfinite(tr.f0)
    voiced_s = float(ok.sum() * (tr.t[1] - tr.t[0])) if len(tr.t) > 1 else 0.0
    obs = Observations(np.nan, voiced_s=voiced_s)
    if voiced_s >= min_voiced_s:
        f0 = float(np.nanmedian(tr.f0))
        # frames close to the median pitch: toots and slides are left out
        sel = ok & (np.abs(1200 * np.log2(np.where(ok, tr.f0, f0) / f0)) < 60)
        f0s = np.where(sel, tr.f0, np.nan)
        n_harm = int(fmax // f0)
        H = harmonic_amplitudes(spec, f0s, n_harm=n_harm, fmax=fmax)[sel]
        with np.errstate(all="ignore"):
            hdb = np.nanpercentile(H, percentile, axis=0)
        n = np.arange(1, n_harm + 1)
        good = np.isfinite(hdb)
        obs.f0 = f0
        obs.harm_f = n[good] * f0
        obs.harm_db = hdb[good]
    decays = find_decays(x, fs)
    if np.isfinite(obs.f0):
        # the reverberation of a played tone decays at its harmonics, not at
        # the bore resonances; drop those modes (and anything below the drone)
        r = np.array([d.freq for d in decays]) / obs.f0
        keep = (r > 0.9) & ((np.abs(1200 * np.log2(np.maximum(r, 1e-3) / np.maximum(np.round(r), 1))) > 50)
                            | (np.round(r) == 1))
        decays = [d for d, k in zip(decays, keep) if k]
    obs.decays = decays
    return obs


# ---------------------------------------------------------------------------
# parametric bore
def geometry(theta, d0=MOUTHPIECE_MM):
    """theta = [length_mm, d(KNOTS[1:]) ...] -> [[x_mm, d_mm], ...]."""
    L = theta[0]
    dk = np.concatenate([[d0], theta[1:len(KNOTS)]])
    u = np.linspace(0, 1, N_SEG + 1)
    d = np.interp(u, KNOTS, dk)
    return [[float(a), float(b)] for a, b in zip(u * L, d)]


TRACT_BOUNDS = [(0.0, 1.0), (0.0, 0.95), (0.1, 1.0), (0.05, 0.9)]   # tongue pos/height, jaw, glottis


def bounds(f0=None):
    # drone 50-110 Hz <-> 1.0-2.2 m; narrower if the drone is known
    if f0 is not None and np.isfinite(f0):
        lmid = 1600.0 * 70.0 / f0
        lb = (0.7 * lmid, 1.35 * lmid)
    else:
        lb = (900.0, 2400.0)
    return [lb] + [(22.0, 70.0), (22.0, 80.0), (22.0, 90.0), (25.0, 100.0), (30.0, 110.0), (35.0, 120.0)]


class _Forward:
    """Fast evaluation of the forward model on the frequencies that the
    observations need."""

    def __init__(self, obs: Observations, params: ModelParams):
        self.obs = obs
        self.params = params
        self.const = compute_moist_air_properties()
        self.hf = obs.harm_f
        self.zvt = vt_impedance(VOWELS["neutral"], self.hf) if len(self.hf) else None
        self.src = np.abs(source(params, len(self.hf))) if len(self.hf) else None
        self.freqs = np.concatenate([PEAK_GRID, self.hf])

    def evaluate(self, geo):
        A, B, C, D, Zr = chain_matrix(Geo(geo), self.freqs, self.const)
        Zin = (A * Zr + B) / (C * Zr + D)
        H = 1.0 / (C * Zr + D)
        n = len(PEAK_GRID)
        return Zin[:n], Zin[n:], H[n:]

    @staticmethod
    def peaks(zgrid):
        m = 20 * np.log10(np.abs(zgrid))
        idx, _ = find_peaks(m)
        idx = idx[(idx > 0) & (idx < len(m) - 1)]
        y0, y1, y2 = m[idx - 1], m[idx], m[idx + 1]
        den = y0 - 2 * y1 + y2
        delta = np.where(np.abs(den) > 1e-9, 0.5 * (y0 - y2) / den, 0.0)
        c = 1200 * np.log2(PEAK_GRID[idx] / PEAK_GRID[0]) + np.clip(delta, -1, 1) * 4.0
        return PEAK_GRID[0] * 2 ** (c / 1200)

    def harmonic_db(self, zin, H, vt=None):
        zs = self.params.zs_mpa * 1e6
        zvt = self.zvt if vt is None else vt_impedance(np.asarray(vt), self.hf)
        G = 1j * self.hf / 100.0 * H * zs / (zs + zin + zvt)
        return 20 * np.log10(np.abs(G) * self.src + 1e-30)


@dataclass
class Weights:
    drone_cents: float = 15.0     # sigma of drone pitch vs. 1st impedance peak
    decay_cents: float = 35.0     # sigma of a decay mode vs. the nearest peak
    decay_min_count: int = 3           # a mode must ring in this many decays (archive: single
                                       # modes are mostly reverberation and fitting artefacts)
    decay_offset_cents: float = -45.0  # free decays ring below the impedance peaks (lips/mouth
                                       # load the mouthpiece; same offset as the toots)
    harm_db: float = 3.0          # sigma of the harmonic-level residual (rms)
    smooth_mm: float = 25.0       # sigma of the second difference of the diameters


def geo_terms(geo, fw: _Forward, w: Weights, vt=None):
    """Misfit terms of a geometry (each ~ chi^2 per observation); None if no peak.
    ``vt``: vocal-tract parameters (4,), default the neutral mouth."""
    zgrid, zin_h, H_h = fw.evaluate(geo)
    pk = fw.peaks(zgrid)
    obs = fw.obs
    terms = {}
    if not len(pk):
        return None, pk
    if np.isfinite(obs.f0):
        terms["drone"] = (1200 * np.log2(obs.f0 / pk[0]) / w.drone_cents) ** 2
    cl = np.array(obs.mode_clusters(min_count=w.decay_min_count))
    if len(cl):
        df, cnt = cl[:, 0], cl[:, 1]
        c = np.min(np.abs(1200 * np.log2(df[:, None] / pk[None, :]) - w.decay_offset_cents), axis=1)
        # robust: a stray room mode must not dominate
        terms["decay"] = float(np.sum(cnt * np.minimum(c / w.decay_cents, 3.0) ** 2) / cnt.sum())
    if len(obs.harm_f) >= 6:
        M = fw.harmonic_db(zin_h, H_h, vt)
        lf = np.log2(obs.harm_f / 1000)
        X = np.stack([np.ones_like(lf), lf], 1)
        Dd = obs.harm_db - M
        res = Dd - X @ np.linalg.lstsq(X, Dd, rcond=None)[0]
        terms["harm"] = float(np.mean(res ** 2)) / w.harm_db ** 2
    return terms, pk


def cost(theta, fw: _Forward, w: Weights, detail=False):
    """theta = bore parameters (see ``geometry``), optionally followed by the
    4 vocal-tract parameters of the mouth shape that is held in the sample."""
    nb = len(KNOTS)
    vt = theta[nb:] if len(theta) > nb else None
    terms, pk = geo_terms(geometry(theta[:nb]), fw, w, vt)
    if terms is None:
        return 1e6
    d = np.concatenate([[MOUTHPIECE_MM], theta[1:nb]])
    terms["smooth"] = float(np.mean(np.diff(d, 2) ** 2)) / w.smooth_mm ** 2
    total = float(sum(terms.values()))
    return (total, terms, pk) if detail else total


# ---------------------------------------------------------------------------
@dataclass
class Estimate:
    obs: Observations
    geometry: list                # best-fitting bore (one of many equivalent ones)
    tract: object                 # fitted mouth shape (VocalTract) or None
    freqs: np.ndarray             # Hz
    zin: np.ndarray               # complex Z_in of the best fit, Pa s/m^3
    band_db: np.ndarray           # (2, N) min/max of 20 log10|Z_in| over the restarts
    resonances: list              # (freq, |Z| MPa, Q) of the best fit
    cost: float
    terms: dict
    restarts: list                # (cost, geometry) of every restart

    def to_dict(self):
        return {
            "drone_hz": self.obs.f0,
            "decay_modes": [vars(d) for d in self.obs.decays],
            "resonances": [{"freq": f, "impedance_mpa": z, "q": q} for f, z, q in self.resonances],
            "cost": self.cost, "terms": self.terms,
            "geometry": self.geometry,
            "tract": vars(self.tract) if self.tract else None,
        }

    @property
    def peaks(self):
        """Impedance maxima as ``Resonance`` (Pa s/m^3), e.g. for ``ModalBore``."""
        return [Resonance(f, z * 1e6, q) for f, z, q in self.resonances]


def estimate(x, fs, params=ModelParams(), weights=Weights(), restarts=3, maxiter=50,
             fmax=3000.0, seed=0, obs=None, progress=None, fit_tract=True):
    """Estimate Z_in(f) of the didgeridoo that produced the sample ``x``.

    With ``fit_tract`` the mouth shape is fitted together with the bore
    (otherwise the neutral mouth is assumed)."""
    obs = obs or observe(x, fs)
    if not np.isfinite(obs.f0) and not obs.decays:
        raise ValueError("no drone and no free decay found in the sample")
    fw = _Forward(obs, params)
    bnds = bounds(obs.f0)
    if fit_tract and len(obs.harm_f) >= 6:
        bnds = bnds + TRACT_BOUNDS
    runs = []
    for r in range(restarts):
        res = differential_evolution(cost, bnds, args=(fw, weights), seed=seed + r,
                                     maxiter=maxiter, popsize=12, tol=1e-6, polish=True)
        runs.append((float(res.fun), res.x))
        if progress:
            progress(r, float(res.fun))
    runs.sort(key=lambda t: t[0])
    best = runs[0][1]
    total, terms, _ = cost(best, fw, weights, detail=True)
    # band: restarts whose cost is close to the best one
    good = [th for c, th in runs if c <= runs[0][0] * 1.5 + 0.5]
    freqs = np.arange(20.0, fmax, 1.0)
    const = fw.const
    curves = []
    for th in good:
        A, B, C, D, Zr = chain_matrix(Geo(geometry(th)), freqs, const)
        curves.append(20 * np.log10(np.abs((A * Zr + B) / (C * Zr + D))))
    curves = np.array(curves)
    geo = geometry(best)
    A, B, C, D, Zr = chain_matrix(Geo(geo), freqs, const)
    zin = (A * Zr + B) / (C * Zr + D)
    b = Bore(geo, fmax=fmax)
    reson = [(r.freq, r.impedance / 1e6, r.q) for r in b.resonances]
    tract = VocalTract(*best[len(KNOTS):]) if len(best) > len(KNOTS) else None
    return Estimate(obs, geo, tract, freqs, zin, np.stack([curves.min(0), curves.max(0)]),
                    reson, total, terms, [(c, geometry(th)) for c, th in runs])


def plot(est: Estimate, path, true_bore=None, title=None, fmax=1500.0):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 4.2))
    m = est.freqs <= fmax
    f = est.freqs[m]
    ax.fill_between(f, est.band_db[0, m] - 120, est.band_db[1, m] - 120, color="C3", alpha=0.2,
                    lw=0, label="restarts")
    ax.plot(f, 20 * np.log10(np.abs(est.zin[m])) - 120, color="C3", label="estimate")
    if true_bore is not None:
        zt = np.abs(true_bore.interp("Zin", f))
        ax.plot(f, 20 * np.log10(zt) - 120, color="k", lw=1, label="measured geometry (DidgeLab)")
    if np.isfinite(est.obs.f0):
        ax.axvline(est.obs.f0, color="C0", ls="--", lw=1, label="drone f0")
    for i, (fm, n) in enumerate(est.obs.mode_clusters(min_count=Weights().decay_min_count)):
        ax.axvline(fm, color="C2", ls=":", lw=1, label="repeated decay modes" if i == 0 else None)
    ax.set_xlim(20, fmax)
    ax.set_xlabel("frequency / Hz")
    ax.set_ylabel("|Z_in| / dB re 1 MPa s m$^{-3}$")
    ax.legend(fontsize=8)
    if title:
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
