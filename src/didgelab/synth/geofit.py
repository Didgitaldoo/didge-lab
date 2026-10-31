"""
Fitting a bore geometry to impedance peaks.

Given a few target peaks (frequency, impedance, Q), find a simple bore whose
simulated impedance spectrum has these peaks. The bore is piecewise conical:
the mouthpiece diameter is fixed, the length and ``n_diameters`` diameters at
equally spaced positions are free. A local least-squares fit (scipy's
``least_squares``) minimises the misfit of the peaks; it starts from a few
typical shapes and adds the target peaks a few at a time, each stage starting
from the best shapes of the previous one.

Many geometries share the same first peaks, so the fitted bore is *one*
didgeridoo with these peaks, not *the* didgeridoo. Rendering it gives a sound
of a physically possible bore (``render_song_loop_from_peaks(...,
method="geometry")``), and a large remaining misfit tells that the target is
not realistic. Fits of up to ~6-8 peaks are reliable; more peaks need more
freedom than this simple bore has (use the optimizer for that).

Example::

    from didgelab.synth import fit_geometry
    fit = fit_geometry([(73.4, 2.0e7, 20), (220.0, 8e6, 25), (330.0, 5e6, 30)])
    fit.geometry          # [[x_mm, d_mm], ...]
    fit.cents             # frequency error of each target peak
"""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares

from ..acoustical_simulation import acoustical_simulation
from ..geo import Geo
from ..sim.sim_interface import compute_moist_air_properties
from .bore import Resonance

PEAK_STEP = 0.002              # log-frequency step of the peak search grid (~3.5 cents)
CENTS_SCALE = 10.0             # 10 cents frequency error counts as much as ...
RATIO_SCALE = np.log(1.25)     # ... a factor 1.25 in impedance or Q (times the weights)
MIN_DIAMETER_MM = 18.0
MAX_DIAMETER_MM = 140.0
# start shapes: (length in units of c / f_drone -- closed-open cylinder 0.25, cone 0.5 --,
#                bell diameter / mouthpiece diameter)
STARTS = [(0.30, 2.0), (0.36, 2.5), (0.30, 1.3), (0.42, 3.0)]


def _as_resonances(peaks):
    res = []
    for p in peaks:
        if isinstance(p, Resonance):
            res.append(p)
        elif isinstance(p, dict):
            res.append(Resonance(float(p["freq"]), float(p["impedance"]), float(p["q"])))
        else:
            res.append(Resonance(*map(float, p)))
    if not res:
        raise ValueError("at least one peak is required")
    for r in res:
        if r.freq <= 0 or r.impedance <= 0 or r.q <= 0:
            raise ValueError(f"freq, impedance and q must be positive: {r}")
    return sorted(res, key=lambda r: r.freq)


def find_peaks(geo, fmax, constants=None, fmin=25.0):
    """Impedance maxima of a geometry below ``fmax`` as :class:`Resonance` list.

    |Z| is simulated on a logarithmic grid; frequency and height of each peak
    come from a parabola through ln|Z| at the three grid points around the
    maximum, Q from its curvature (for a resonance, ln|Z| ~ ln Z_n - 4 Q^2
    (f - f_n)^2 / f_n^2 near the peak).
    """
    constants = constants or compute_moist_air_properties()
    f = np.exp(np.arange(np.log(fmin), np.log(fmax), PEAK_STEP))
    lz = np.log(np.asarray(acoustical_simulation(Geo(geo), f, constants=constants)).ravel())
    idx = np.where((lz[1:-1] > lz[:-2]) & (lz[1:-1] >= lz[2:]))[0] + 1
    out = []
    for k in idx:
        y0, y1, y2 = lz[k - 1], lz[k], lz[k + 1]
        den = y0 - 2 * y1 + y2
        d = 0.5 * (y0 - y2) / den if den < 0 else 0.0
        freq = f[k] * np.exp(PEAK_STEP * d)
        curv = den / (f[k] * PEAK_STEP) ** 2          # d^2 ln|Z| / df^2
        q = freq * np.sqrt(max(-curv, 1e-12) / 4)
        out.append(Resonance(float(freq), float(np.exp(y1 - 0.25 * (y0 - y2) * d)), float(q)))
    return out


def bore_geometry(x, mouth_diameter):
    """Parameter vector [length_m, ln d_1, ..., ln d_K] -> [[x_mm, d_mm], ...]."""
    L = x[0] * 1000.0
    xs = np.linspace(0.0, L, len(x))
    return [[0.0, float(mouth_diameter)]] + [[float(a), float(np.exp(b))] for a, b in zip(xs[1:], x[1:])]


@dataclass
class GeometryFit:
    geometry: list          # [[x_mm, d_mm], ...]
    targets: list           # target peaks (Resonance), sorted by frequency
    peaks: list             # peaks of the fitted geometry matched to the targets (Resonance or None)
    nfev: int               # number of simulations

    @property
    def cents(self):
        """Frequency error of each target peak in cents (NaN if the bore has no such peak)."""
        return np.array([1200 * np.log2(p.freq / t.freq) if p else np.nan
                         for p, t in zip(self.peaks, self.targets)])

    @property
    def impedance_ratio(self):
        return np.array([p.impedance / t.impedance if p else np.nan for p, t in zip(self.peaks, self.targets)])

    @property
    def q_ratio(self):
        return np.array([p.q / t.q if p else np.nan for p, t in zip(self.peaks, self.targets)])

    @property
    def max_cents(self):
        """Largest frequency error (inf if a target peak is missing)."""
        c = np.abs(self.cents)
        return float(np.inf if np.isnan(c).any() else c.max())


def fit_geometry(peaks, mouth_diameter=32.0, n_diameters=12, temperature=28.0, constants=None,
                 impedance_weight=0.3, q_weight=0.3, stage_size=3, max_nfev=300):
    """Fit a bore to target impedance peaks.

    Parameters
    ----------
    peaks : (freq_hz, impedance, q) tuples, dicts with keys freq, impedance, q,
        or :class:`Resonance`; impedance in Pa s/m^3 as computed by DidgeLab
    mouth_diameter : mouthpiece diameter in mm (fixed)
    n_diameters : number of free diameters along the bore
    temperature : air temperature in degrees Celsius (ignored if ``constants`` is given)
    impedance_weight, q_weight : weight of the impedance and Q misfit relative
        to the frequency misfit (10 cents ~ a factor 1.25 at weight 1)
    stage_size : number of target peaks added per stage
    max_nfev : simulations per least-squares run

    Returns a :class:`GeometryFit`.
    """
    targets = _as_resonances(peaks)
    constants = constants or compute_moist_air_properties(temp_celsius=temperature)
    ft = np.array([t.freq for t in targets])
    zt = np.array([t.impedance for t in targets])
    qt = np.array([t.q for t in targets])
    K = int(n_diameters)
    lb = np.r_[0.6, np.full(K, np.log(MIN_DIAMETER_MM))]
    ub = np.r_[3.5, np.full(K, np.log(MAX_DIAMETER_MM))]

    def residuals(n_use):
        fmax = ft[n_use - 1] * 1.15

        def res(x):
            pk = find_peaks(bore_geometry(x, mouth_diameter), fmax, constants)
            r = np.empty(3 * n_use)
            for n in range(n_use):
                if n < len(pk):
                    r[3 * n] = 1200 * np.log2(pk[n].freq / ft[n]) / CENTS_SCALE
                    r[3 * n + 1] = impedance_weight * np.log(pk[n].impedance / zt[n]) / RATIO_SCALE
                    r[3 * n + 2] = q_weight * np.log(pk[n].q / qt[n]) / RATIO_SCALE
                else:   # the bore has fewer peaks than the target
                    r[3 * n:3 * n + 3] = (100.0, 10.0, 10.0)
            return r
        return res

    c = constants.speed_of_sound
    rel = np.arange(1, K + 1) / K
    cand = [np.clip(np.r_[lf * c / ft[0], np.log(mouth_diameter * (1 + (bell - 1) * rel ** 2))],
                    lb + 1e-6, ub - 1e-6) for lf, bell in STARTS]
    n_use, nfev = min(stage_size, len(ft)), 0
    while True:
        res = residuals(n_use)
        out = [least_squares(res, x, bounds=(lb, ub), diff_step=1e-3, max_nfev=max_nfev) for x in cand]
        nfev += sum(o.nfev + o.njev * len(x) for o, x in zip(out, cand))   # + finite differences
        out.sort(key=lambda o: o.cost)
        cand = [o.x for o in out[:2]]
        if n_use == len(ft):
            break
        n_use = min(len(ft), n_use + stage_size)

    geo = bore_geometry(cand[0], mouth_diameter)
    pk = find_peaks(geo, ft[-1] * 1.15, constants)
    matched = [pk[n] if n < len(pk) else None for n in range(len(ft))]
    return GeometryFit(geometry=geo, targets=targets, peaks=matched, nfev=nfev)
