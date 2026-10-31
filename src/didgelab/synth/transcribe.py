"""
Transcription of playing techniques in a recording.

Combines the frame analysis (``detect.analyse``) with the learned detector
(``learn.Detector``, trained on synthetic performances). Without a trained
model the rule-based ``detect.detect`` is used.

Ported from didge-synth (``didgesynth.transcribe``).
"""

from pathlib import Path

import numpy as np

from .detect import analyse, detect
from .model import ModelParams

DEFAULT_MODEL = Path(__file__).parent / "data" / "detector.joblib"
VOICE_SIDEBAND_DB = -20.0   # minimum level of the sum/difference tones of a vocalisation
TRILL_AM_DB = 1.5           # minimum 15-45 Hz amplitude modulation of a trill

COLORS = {"stop_front": "#1f77b4", "stop_back": "#ff7f0e", "trill": "#2ca02c", "voice": "#d62728",
          "circular_breath": "#9467bd", "fricative": "#8c564b", "toot": "#e377c2"}


def transcribe(x, fs, bore, model_path=None, params=ModelParams()):
    fa = analyse(x, fs, bore, params)
    path = Path(model_path) if model_path else DEFAULT_MODEL
    try:
        from .learn import Detector, features, events_from_proba
        det = Detector.load(path) if path.exists() else None
    except ImportError:          # scikit-learn / joblib not installed
        det = None
    except Exception as e:       # e.g. a model saved with an incompatible scikit-learn version
        import warnings
        warnings.warn(f"cannot load the technique detector {path} ({e}); using the rule-based "
                      "detector. Retrain it with scripts/synth/train_detector.py.")
        det = None
    if det is not None:
        P = det.predict_proba(features(fa, bore))
        segs = events_from_proba(P, fa.t, det.thresholds, fa)
        # physical plausibility checks: a vocalisation must show sum and
        # difference tones between the harmonics, a trill a 20-40 Hz modulation
        from .analysis import sideband_level
        keep = []
        for s in segs:
            if s.label == "voice":
                i0, i1 = int(s.start * fs), int(s.end * fs)
                f0 = np.nanmedian(fa.f0[(fa.t >= s.start) & (fa.t <= s.end)])
                lev = sideband_level(x[i0:i1], fs, f0)
                s.info["sideband_db"] = lev
                if lev < VOICE_SIDEBAND_DB:
                    continue
            if s.label == "trill":
                # a rolled r modulates the sound periodically at 20-40 Hz
                m = (fa.t >= s.start) & (fa.t <= s.end)
                am = float(np.median(fa.am_depth[m])) if m.any() else 0.0
                s.info["am_depth_db"] = am
                if am < TRILL_AM_DB:
                    continue
            keep.append(s)
        segs = keep
        for s in segs:
            if s.label.startswith("stop"):
                i = int(np.argmin(np.abs(fa.t - (s.start + s.end) / 2)))
                ti = fa.tongue_index[max(0, i - 5):i + 5]
                s.info["tongue_index"] = float(np.nanmedian(ti)) if np.any(np.isfinite(ti)) else None
        fa.proba = P
    else:
        segs = detect(fa)
    return fa, segs


def plot_transcription(x, fs, fa, segs, path, title=""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from .analysis import stft

    t, f, X = stft(x, fs, n_fft=4096, win=2048, hop=480)
    S = 20 * np.log10(X + 1e-9)
    S -= S.max()
    m = f < 4000
    fig, axs = plt.subplots(4, 1, figsize=(16, 11), sharex=True,
                            gridspec_kw={"height_ratios": [3, 1, 1.3, 1]})
    axs[0].pcolormesh(t, f[m], S[:, m].T, vmin=-75, vmax=0, shading="auto", cmap="magma")
    axs[0].plot(fa.t, fa.f0, color="cyan", lw=0.8)
    axs[0].set_ylabel("Hz")
    axs[0].set_title(title)
    axs[1].plot(fa.t, fa.level_db, color="k", lw=0.8)
    axs[1].set_ylabel("level dB")
    if fa.formant_hz is not None:
        axs[0].plot(fa.t, fa.formant_hz, ".", color="white", ms=1.5, alpha=0.7)
        axs[2].plot(fa.t, fa.tongue_index, color="#1f77b4", lw=0.9)
        axs[2].set_ylabel("tongue index\n0 front .. 1 back", fontsize=8)
        axs[2].set_ylim(-0.3, 1.3)
    rows = ["toot", "stop_front", "stop_back", "trill", "voice", "circular_breath", "fricative"]
    for s in segs:
        key = "toot" if s.label.startswith("toot") else s.label
        if key not in rows:
            continue
        y = rows.index(key)
        axs[3].broken_barh([(s.start, max(s.end - s.start, 0.02))], (y - 0.4, 0.8),
                           color=COLORS.get(key, "gray"))
    axs[3].set_yticks(range(len(rows)))
    axs[3].set_yticklabels(rows, fontsize=8)
    axs[3].set_xlabel("time / s")
    plt.tight_layout()
    plt.savefig(path, dpi=65)
    plt.close(fig)


def resynthesize(fa, bore, params=ModelParams(), fs=48000, rate=500.0, seed=0, reverb=0.15):
    """Copy synthesis: render the analysed trajectories.

    f0 and level are taken from the analysis; the mouth shape follows the
    measured formant: the tongue index is mapped back onto the tongue-position
    axis of the vocal-tract model. Vocalisation and noises are not
    resynthesised.
    """
    from scipy.ndimage import median_filter
    from .synth import Controls, Synthesizer
    from .vocaltract import VOWELS

    dur = fa.t[-1] + (fa.t[1] - fa.t[0])
    c = Controls.constant(dur, rate, bore.drone, params, VOWELS["neutral"])
    tc = np.arange(c.n) / rate
    ok = np.isfinite(fa.f0)

    def fill(v, default):
        v = np.asarray(v, float).copy()
        good = np.isfinite(v)
        if not good.any():
            return np.full(len(v), default)
        return np.interp(np.arange(len(v)), np.where(good)[0], v[good])

    c.f0 = np.interp(tc, fa.t, median_filter(fill(fa.f0, bore.drone), 5))
    lev = np.where(ok, fa.level_db, np.nanmin(fa.level_db) - 20)
    amp = 10 ** ((lev - np.nanmax(lev)) / 20)
    c.amp = np.interp(tc, fa.t, median_filter(amp, 3))
    # mouth shape from the formant (articulating mouth: nearly closed jaw);
    # frames without a clear formant keep the last tongue position
    ti = fill(fa.tongue_index, 0.5)
    vt = np.stack([0.4 + 0.55 * np.clip(ti, 0, 1), np.full_like(ti, 0.5),
                   np.full_like(ti, 0.45), np.full_like(ti, 0.3)], 1)
    for k in range(4):
        c.vt[:, k] = np.interp(tc, fa.t, median_filter(vt[:, k], 5))
    syn = Synthesizer(bore, params, fs=fs, seed=seed)
    return syn.render(c, reverb=reverb)
