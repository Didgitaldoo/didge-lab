"""
Learning the technique detector from synthetic performances.

The synthesizer is used as a data generator: random scores are played by
random players on random bores, rendered with random recording conditions,
and analysed with ``detect.analyse``. Because the controls are known, every
analysis frame has ground-truth labels. Gradient-boosted trees (one per
technique) are trained on frame features with +-100 ms context.

Evaluation uses *unseen instruments*: the detector is trained on random
parametric bores only and tested on the Didge Archive geometries.

Ported from didge-synth (``didgesynth.learn``).
"""

import json
import os
from dataclasses import dataclass
from multiprocessing import Pool

import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import find_peaks

from .bore import Bore
from .detect import analyse, Segment, _runs
from .model import ModelParams
from .synth import Synthesizer
from .techniques import random_score, random_player, parse_score

CLASSES = ["stop_front", "stop_back", "trill", "voice", "circular_breath", "fricative"]
LABEL_MAP = {"stop_front": "stop_front", "stop_front_voiced": "stop_front",
             "stop_back": "stop_back", "stop_back_voiced": "stop_back",
             "trill": "trill", "voice": "voice", "circular_breath": "circular_breath",
             "fricative": "fricative", "aspirate": "fricative"}
CONTEXT = [-10, -5, -3, -1, 0, 1, 3, 5, 10]


# ---------------------------------------------------------------------------
def random_geometry(rng):
    """A random but plausible didgeridoo bore (list of [x_mm, d_mm])."""
    L = rng.uniform(1150, 1800)
    d0 = rng.uniform(28, 34)
    d1 = rng.uniform(55, 100)
    expo = rng.uniform(0.6, 3.0)
    x = np.linspace(0, L, int(rng.integers(14, 26)))
    d = d0 + (d1 - d0) * (x / L) ** expo
    for _ in range(int(rng.integers(0, 4))):
        c, w, h = rng.uniform(0.1, 0.85) * L, rng.uniform(40, 200), rng.uniform(-6, 9)
        d += h * np.exp(-0.5 * ((x - c) / w) ** 2)
    d = np.maximum(d, 22)
    return [[float(a), float(b)] for a, b in zip(x, d)]


def frame_labels(gestures, t):
    Y = np.zeros((len(t), len(CLASSES)), dtype=np.int8)
    for s, e, lab in gestures:
        cls = LABEL_MAP.get(lab)
        if cls is None:
            continue
        Y[(t >= s) & (t < e), CLASSES.index(cls)] = 1
    return Y


def base_features(fa, bore):
    lev = np.nan_to_num(fa.level_db, nan=np.nanmin(fa.level_db))
    ref = np.nanmedian(lev[fa.mode >= 0]) if np.any(fa.mode >= 0) else np.median(lev)
    lev = lev - ref
    run = median_filter(lev, size=25, mode="nearest")
    res = np.array([r.freq for r in bore.resonances[:6]])
    cents = np.zeros(len(fa.t))
    ok = fa.mode >= 0
    cents[ok] = 1200 * np.log2(fa.f0[ok] / res[fa.mode[ok]])
    feats = [lev, lev - run, np.gradient(lev), np.gradient(np.gradient(lev)),
             (fa.mode == 0).astype(float), (fa.mode >= 1).astype(float), cents,
             np.nan_to_num(fa.inharm, nan=0.5), np.nan_to_num(fa.hf_noise_db, nan=0.0),
             fa.am_depth, np.nan_to_num(fa.am_rate, nan=0.0), np.nan_to_num(fa.fit_err, nan=10.0)]
    feats += [np.nan_to_num(fa.vt[:, k], nan=-1) for k in range(4)]
    if fa.formant_hz is not None:
        lf = np.nan_to_num(np.log2(fa.formant_hz / 1000.0), nan=-3.0)
        feats += [lf, np.gradient(lf), np.nan_to_num(fa.formant_prom, nan=0.0),
                  np.nan_to_num(fa.tongue_index, nan=-1.0)]
    B = np.nan_to_num(fa.bands_db, nan=-60.0)
    feats += [B[:, k] for k in range(B.shape[1])]
    feats += [np.gradient(B[:, k]) for k in range(B.shape[1])]
    F = np.stack(feats, 1)
    return F


def context_features(F):
    T = len(F)
    out = []
    for o in CONTEXT:
        idx = np.clip(np.arange(T) + o, 0, T - 1)
        out.append(F[idx])
    # local statistics over +-150 ms
    from scipy.ndimage import uniform_filter1d
    out.append(uniform_filter1d(F, 31, axis=0, mode="nearest"))
    out.append(np.sqrt(np.maximum(uniform_filter1d(F ** 2, 31, axis=0, mode="nearest")
                                  - uniform_filter1d(F, 31, axis=0, mode="nearest") ** 2, 0)))
    return np.concatenate(out, 1).astype(np.float32)


def features(fa, bore):
    return context_features(base_features(fa, bore))


# ---------------------------------------------------------------------------
def make_example(args):
    """Render and analyse one random performance; returns X, Y, meta."""
    seed, geometry, duration = args
    rng = np.random.default_rng(seed)
    geo = geometry if geometry is not None else random_geometry(rng)
    bore = Bore(geo)
    player = random_player(rng, bore)
    score = random_score(rng, duration, n_toots=len(bore.resonances))
    events, settings = parse_score(score)
    c = player.controls(events, settings)
    syn = Synthesizer(bore, player.params, seed=int(rng.integers(1 << 30)))
    y = syn.render(c, reverb=float(rng.uniform(0.0, 0.4)))
    y = degrade(y, rng)
    fa = analyse(y, 48000, bore, ModelParams())
    X = features(fa, bore)
    Y = frame_labels(c.gestures, fa.t)
    toots = [(s, e, lab) for s, e, lab in c.gestures if lab.startswith("toot")]
    meta = {"seed": seed, "score": score, "gestures": c.gestures, "toots": toots,
            "mode": fa.mode.tolist(), "t": fa.t.tolist(), "level_db": fa.level_db.tolist()}
    return X, Y, meta


def degrade(y, rng, fs=48000):
    """Recording conditions: microphone tilt, background noise, gain."""
    n = len(y)
    Y = np.fft.rfft(y)
    f = np.fft.rfftfreq(n, 1 / fs)
    tilt = rng.uniform(-3, 3)
    Y *= 10 ** (tilt * np.log2(np.maximum(f, 20) / 1000) / 20)
    y = np.fft.irfft(Y, n)
    noise = np.fft.irfft(np.fft.rfft(rng.standard_normal(n)) / np.sqrt(np.maximum(f, 20) / 20), n)
    snr = rng.uniform(30, 55)
    y = y + noise / np.std(noise) * np.std(y) * 10 ** (-snr / 20)
    return y / np.max(np.abs(y)) * rng.uniform(0.3, 0.95)


def build_dataset(jobs, processes=3):
    with Pool(processes) as pool:
        return pool.map(make_example, jobs)


# ---------------------------------------------------------------------------
@dataclass
class Detector:
    models: list
    thresholds: list

    def predict_proba(self, X):
        return np.stack([m.predict_proba(X)[:, 1] for m in self.models], 1)

    def save(self, path):
        import joblib
        joblib.dump({"models": self.models, "thresholds": self.thresholds, "classes": CLASSES}, path,
                    compress=3)

    @classmethod
    def load(cls, path):
        import joblib
        d = joblib.load(path)
        return cls(d["models"], d["thresholds"])


def train(examples, max_iter=250):
    from sklearn.ensemble import HistGradientBoostingClassifier
    X = np.concatenate([e[0] for e in examples])
    Y = np.concatenate([e[1] for e in examples])
    models = []
    for k, name in enumerate(CLASSES):
        m = HistGradientBoostingClassifier(max_iter=max_iter, learning_rate=0.08, max_leaf_nodes=48,
                                           l2_regularization=1.0, class_weight="balanced",
                                           random_state=k)
        m.fit(X, Y[:, k])
        models.append(m)
    return Detector(models, [0.5] * len(CLASSES))


# ---------------------------------------------------------------------------
def events_from_proba(P, t, thresholds, fa=None):
    """Turn frame probabilities into event segments.

    Only frames in which the instrument sounds (lip mode found, level within
    35 dB of the maximum) can carry an event.
    """
    dt = t[1] - t[0]
    segs = []
    if fa is not None and getattr(fa, "level_db", None) is not None:
        lev = np.nan_to_num(fa.level_db, nan=-200.0)
        sounding = (np.asarray(fa.mode) >= 0) & (lev > np.max(lev) - 35)
        P = P * sounding[:, None]
    # stops: one event per run of p(front)+p(back) above threshold; long runs
    # (fast articulation) are split at maxima at least 100 ms apart
    ps = median_filter(P[:, 0] + P[:, 1], size=3)
    pk = []
    for i, j in _runs(ps > thresholds[0], t, 0.02):
        if t[j - 1] - t[i] < 0.12:
            pk.append(i + int(np.argmax(ps[i:j])))
        else:
            q, _ = find_peaks(np.concatenate([[0], ps[i:j], [0]]), distance=int(0.1 / dt))
            pk += [i + k - 1 for k in q]
    for i in pk:
        front = P[i, 0] >= P[i, 1]
        segs.append(Segment(t[i] - 0.02, t[i] + 0.02, "stop_front" if front else "stop_back",
                            {"p": float(ps[i])}))
    min_len = {"trill": 0.08, "voice": 0.15, "circular_breath": 0.1, "fricative": 0.04}
    for k, name in enumerate(CLASSES[2:], 2):
        p = median_filter(P[:, k], size=7)
        for i, j in _runs(p > thresholds[k], t, min_len[name]):
            segs.append(Segment(t[i], t[j - 1] + dt, name, {"p": float(p[i:j].mean())}))
    if fa is not None:
        for i, j in _runs(fa.mode >= 1, t, 0.08):
            k = int(np.median(fa.mode[i:j]))
            f0 = fa.f0[i:j]
            segs.append(Segment(t[i], t[j - 1] + dt, f"toot{k}",
                                {"f0": float(np.nanmedian(f0)) if np.any(np.isfinite(f0)) else None}))
    segs.sort(key=lambda s: s.start)
    return segs


def match_events(pred, truth, tol=0.06, point=False):
    """Greedy matching. point=True: centres within tol; else any overlap."""
    used = set()
    tp = 0
    for p in pred:
        for j, (s, e) in enumerate(truth):
            if j in used:
                continue
            if point:
                ok = abs((p.start + p.end) / 2 - (s + e) / 2) <= tol
            else:
                ok = p.start < e and p.end > s
            if ok:
                used.add(j)
                tp += 1
                break
    return tp, len(pred), len(truth)


def _merge(intervals, gap=0.05):
    """Merge touching ground-truth intervals of the same class (e.g. two
    consecutive vocalised syllables form one vocalisation)."""
    out = []
    for s, e in sorted(intervals):
        if out and s <= out[-1][1] + gap:
            out[-1] = (out[-1][0], max(out[-1][1], e))
        else:
            out.append((s, e))
    return out


def evaluate_events(examples_meta, segs_list):
    """Event precision/recall per technique on synthetic examples."""
    stats = {}
    for meta, segs in zip(examples_meta, segs_list):
        G = meta["gestures"]
        groups = {
            "stop": ([s for s in segs if s.label.startswith("stop")],
                     [(s, e) for s, e, l in G if l.startswith("stop")], True),
            "trill": ([s for s in segs if s.label == "trill"], [(s, e) for s, e, l in G if l == "trill"], False),
            "voice": ([s for s in segs if s.label == "voice"], [(s, e) for s, e, l in G if l == "voice"], False),
            "circular_breath": ([s for s in segs if s.label == "circular_breath"],
                                [(s, e) for s, e, l in G if l == "circular_breath"], False),
            "fricative": ([s for s in segs if s.label == "fricative"],
                          [(s, e) for s, e, l in G if LABEL_MAP.get(l) == "fricative"], False),
            "toot": ([s for s in segs if s.label.startswith("toot")],
                     [(s, e) for s, e, l in G if l.startswith("toot")], False),
        }
        for name, (pred, truth, point) in groups.items():
            if not point:
                truth = _merge(truth)
            tp, npred, ntrue = match_events(pred, truth, point=point)
            st = stats.setdefault(name, [0, 0, 0])
            st[0] += tp
            st[1] += npred
            st[2] += ntrue
        # place of articulation of matched stops
        pl = stats.setdefault("stop_place", [0, 0])
        for p in [s for s in segs if s.label.startswith("stop")]:
            c = (p.start + p.end) / 2
            for s, e, l in G:
                if l.startswith("stop") and abs(c - (s + e) / 2) <= 0.06:
                    pl[1] += 1
                    pl[0] += int(l.split("_")[1] == p.label.split("_")[1])
                    break
    out = {}
    for name, st in stats.items():
        if name == "stop_place":
            out[name] = {"accuracy": st[0] / max(st[1], 1), "n": st[1]}
            continue
        tp, npred, ntrue = st
        prec = tp / max(npred, 1)
        rec = tp / max(ntrue, 1)
        out[name] = {"precision": prec, "recall": rec,
                     "f1": 2 * prec * rec / max(prec + rec, 1e-9), "n_true": ntrue, "n_pred": npred}
    return out
