"""Evaluate the realism of the synthesizer against the Didge Archive.

    python scripts/synth/evaluate_archive.py [--archive PATH]

Writes doc/results/synth/archive_eval.json, figures to doc/figures/ and synthesized
neutral drones to outputs/synth-audio/.
"""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np
from scipy.signal import find_peaks, butter, freqz

from didgelab.synth.cli import write_audio
from didgelab.synth import archive
from didgelab.synth.analysis import load_audio, stft, track_f0, harmonic_amplitudes, ltas_harmonics
from didgelab.synth.bore import Bore
from didgelab.synth.excitation import hindle_harmonics
from didgelab.synth.model import ModelParams, harmonic_db, filter_gain
from didgelab.synth.synth import Synthesizer
from didgelab.synth.techniques import score_controls
from didgelab.synth.vocaltract import VOWELS

ap = argparse.ArgumentParser()
ap.add_argument("--archive", default=None)
args = ap.parse_args()

FIG = Path("doc/figures")
FIG.mkdir(parents=True, exist_ok=True)
AUD = Path("outputs/synth-audio")
AUD.mkdir(parents=True, exist_ok=True)
Path("doc/results/synth").mkdir(parents=True, exist_ok=True)

insts = archive.load(args.archive)
bores = {i.name: Bore(i.geometry) for i in insts}
out = {}


def cents(a, b):
    return 1200 * np.log2(a / b)


# ---------------------------------------------------------------- A: drone
rows = []
neutral = {}
for ins in insts:
    x, fs = load_audio(ins.samples["neutral-sound"], 48000)
    spec = stft(x, fs)
    tr = track_f0(x, fs, spec=spec)
    f0 = float(np.nanmedian(tr.f0))
    H = harmonic_amplitudes(spec, tr.f0, n_harm=80, fmax=6000)
    neutral[ins.name] = (f0, ltas_harmonics(H), x)
    rows.append({"instrument": ins.name, "sim_drone_hz": bores[ins.name].drone, "rec_drone_hz": f0,
                 "error_cents": float(cents(f0, bores[ins.name].drone))})
errs = [r["error_cents"] for r in rows]
out["drone"] = {"rows": rows, "mean_abs_cents": float(np.mean(np.abs(errs))),
                "mean_cents": float(np.mean(errs))}
print("drone", out["drone"]["mean_abs_cents"], flush=True)


# ---------------------------------------------------------------- A2: toots in the songs
# Pitch clusters above the drone in the song recordings (generic tracker, no
# resonance prior). A cluster is attributed to a toot only if it is closer to
# a simulated resonance than to a harmonic of the drone (otherwise a strong
# harmonic or a formant could be mistaken for a toot), and at least
# HARMONIC_MARGIN cents away from that harmonic (otherwise it is ambiguous).
HARMONIC_MARGIN = 50.0
toot_rows = []
for ins in insts:
    if "song" not in ins.samples:
        continue
    b = bores[ins.name]
    res = np.array([r.freq for r in b.resonances[:6]])
    x, fs = load_audio(ins.samples["song"], 48000)
    tr = track_f0(x, fs, fmin=0.8 * res[0], fmax=1.1 * res[-1], sub_margin=2.0)
    ok = np.isfinite(tr.f0) & (tr.score > 15)
    c = 1200 * np.log2(tr.f0[ok] / res[0])
    h, e = np.histogram(c, bins=np.arange(150, 3300, 25))
    for i in range(len(h)):
        if h[i] >= 8 and h[i] == h[max(0, i - 3):i + 4].max():
            cc = e[i] + 12.5
            sel = np.abs(c - cc) < 60
            f_rec = float(np.median(tr.f0[ok][sel]))
            k = int(np.argmin(np.abs(cents(f_rec, res[1:])))) + 1
            harm = round(f_rec / res[0])
            d_res = float(cents(f_rec, res[k]))
            d_harm = float(cents(f_rec, harm * res[0]))
            toot_rows.append({"instrument": ins.name, "rec_hz": f_rec, "frames": int(h[i]), "resonance": k,
                              "sim_hz": float(res[k]), "error_cents": d_res, "nearest_harmonic": int(harm),
                              "harmonic_cents": d_harm,
                              "is_toot": abs(d_res) < abs(d_harm) and abs(d_harm) > HARMONIC_MARGIN})
toots = [r for r in toot_rows if r["is_toot"]]
te = np.array([r["error_cents"] for r in toots])
out["toots"] = {"clusters": toot_rows, "n_toot_clusters": len(toots),
                "mean_cents": float(np.mean(te)) if len(te) else None,
                "mean_abs_cents": float(np.mean(np.abs(te))) if len(te) else None}
print("toots", out["toots"]["mean_cents"], len(toots), flush=True)

# ---------------------------------------------------------------- A3: hand slaps
slap_rows = []
for ins in insts:
    b = bores[ins.name]
    mag = np.abs(b.Zin)
    lo = b.freqs < 1000
    maxima = b.freqs[lo][find_peaks(mag[lo])[0]][:4]
    minima = b.freqs[lo][find_peaks(-mag[lo])[0]][:4]
    for key, targets in [("hand-closed", maxima), ("hand-open", minima)]:
        if key not in ins.samples:
            continue
        x, fs = load_audio(ins.samples[key])
        N = len(x)
        X = np.abs(np.fft.rfft(x * np.hanning(N), 16 * N))
        f = np.fft.rfftfreq(16 * N, 1 / fs)
        for tgt in targets:
            m = (f > tgt * 2 ** (-150 / 1200)) & (f < tgt * 2 ** (150 / 1200))
            pk = find_peaks(X[m])[0]
            if not len(pk):
                continue
            fp = f[m][pk[np.argmax(X[m][pk])]]
            slap_rows.append({"instrument": ins.name, "kind": key, "sim_hz": float(tgt), "rec_hz": float(fp),
                              "error_cents": float(cents(fp, tgt))})
for key in ["hand-closed", "hand-open"]:
    e = np.array([r["error_cents"] for r in slap_rows if r["kind"] == key])
    out[f"slap_{key}"] = {"n": len(e), "median_abs_cents": float(np.median(np.abs(e))),
                          "within_50_cents": float(np.mean(np.abs(e) < 50))}
out["slap_rows"] = slap_rows
print("slaps", out["slap_hand-closed"], out["slap_hand-open"], flush=True)

# ---------------------------------------------------------------- B: harmonic fingerprint
NL = 10
n = np.arange(1, NL + 1)
names = [i.name for i in insts]
params = ModelParams()


def pattern(bname, f0, p=params, vt=VOWELS["neutral"]):
    M = harmonic_db(bores[bname], f0, vt, p, NL)
    return M - M[0]


def pair_dist(L, M):
    r = (L[1:NL] - L[0]) - M[1:]
    X = np.stack([np.log2(n[1:]), np.ones(NL - 1)], 1)
    c, *_ = np.linalg.lstsq(X, r, rcond=None)
    return float(np.sqrt(np.mean((r - X @ c) ** 2)))


def fingerprint(model_fn):
    D = np.zeros((len(names), len(names)))
    for i, a in enumerate(names):
        f0, L, _ = neutral[a]
        for j, b in enumerate(names):
            D[i, j] = pair_dist(L, model_fn(b, f0))
    same = {i: [j for j, b in enumerate(names) if insts[j].geometry_file == insts[i].geometry_file]
            for i in range(len(names))}
    acc = np.mean([np.argmin(D[i]) in same[i] for i in range(len(names))])
    rank = np.mean([np.sum(D[i] < min(D[i, j] for j in same[i])) for i in range(len(names))])
    return D, float(acc), float(rank)


D, acc, rank = fingerprint(pattern)


def d2w_pattern(bname, f0):
    b = bores[bname]
    M = 20 * np.log10(np.abs(b.interp("Zin", n * f0)) ** 1.2 * np.abs(hindle_harmonics(0.85, NL)) + 1e-30)
    return M - M[0]


D2, acc2, rank2 = fingerprint(d2w_pattern)
out["fingerprint"] = {"model": {"accuracy": acc, "mean_rank": rank, "diag_rms_db": float(np.mean(np.diag(D)))},
                      "didge2wav": {"accuracy": acc2, "mean_rank": rank2, "diag_rms_db": float(np.mean(np.diag(D2)))},
                      "chance": 1 / len(set(i.geometry_file for i in insts)), "matrix": D.tolist(), "names": names}
print("fingerprint", acc, acc2, flush=True)

# ---------------------------------------------------------------- C: third-octave LTAS, leave-one-out
NH = 80
edges = 63 * 2 ** (np.arange(0, 20) / 3)


def bands(f, Ldb):
    P = 10 ** (Ldb / 10)
    return np.array([10 * np.log10(P[(f >= lo) & (f < hi)].sum()) if np.any((f >= lo) & (f < hi)) else np.nan
                     for lo, hi in zip(edges[:-1], edges[1:])])


def band_err(rb, mb):
    ok = np.isfinite(rb) & np.isfinite(mb)
    d = rb[ok] - mb[ok]
    return float(np.sqrt(np.mean((d - d.mean()) ** 2)))


nn = np.arange(1, NH + 1)
RB = {nm: bands(nn * neutral[nm][0], neutral[nm][1]) for nm in names}
grid = list(itertools.product([0.3, 0.45, 0.6], [0.7, 0.85], [0.35, 1.0, 2.0]))
E = {}
for oq, sk, zs in grid:
    p = ModelParams(oq=oq, skew=sk, zs_mpa=zs)
    for nm in names:
        f0 = neutral[nm][0]
        E[(oq, sk, zs, nm)] = band_err(RB[nm], bands(nn * f0, harmonic_db(bores[nm], f0, VOWELS["neutral"], p, NH)))
loo = []
for nm in names:
    best = min(grid, key=lambda g: np.mean([E[(*g, o)] for o in names if o != nm]))
    loo.append(E[(*best, nm)])
bb, aa = butter(2, 2500 / 24000)
d2w, flat, default = [], [], []
for nm in names:
    f0 = neutral[nm][0]
    b = bores[nm]
    _, hl = freqz(bb, aa, worN=nn * f0, fs=48000)
    M = 20 * np.log10(np.abs(b.interp("Zin", nn * f0)) ** 1.2 * np.abs(hindle_harmonics(0.85, NH)) * np.abs(hl) + 1e-30)
    d2w.append(band_err(RB[nm], bands(nn * f0, np.maximum(M, M.max() - 120))))
    flat.append(band_err(RB[nm], bands(nn * f0, 20 * np.log10(1 / nn))))
    # the final defaults of ModelParams (not a grid point: zs_mpa = 0.1)
    default.append(band_err(RB[nm], bands(nn * f0, harmonic_db(b, f0, VOWELS["neutral"], ModelParams(), NH))))
out["ltas_third_octave"] = {"model_leave_one_out_db": float(np.mean(loo)), "model_default_db": float(np.mean(default)),
                            "didge2wav_db": float(np.mean(d2w)), "generic_1_over_n_db": float(np.mean(flat)),
                            "per_instrument_default": dict(zip(names, default))}
print("ltas", out["ltas_third_octave"], flush=True)

# ---------------------------------------------------------------- E: synthesized drones + figure
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, axs = plt.subplots(3, 3, figsize=(15, 10))
for ax, ins in zip(axs.ravel(), insts):
    b = bores[ins.name]
    f0, L, x = neutral[ins.name]
    c = score_controls(b, "tempo = 60\nuh:2", params)
    y = Synthesizer(b, params, seed=0).render(c, reverb=0.15)
    write_audio(AUD / f"{ins.name.lower().replace(' ', '-')}-neutral-synth.mp3", y, 48000)
    for sig, lab, col in [(x, "recording", "#555"), (y[int(0.3 * 48000):int(1.8 * 48000)], "synthesis", "#d62728")]:
        N = len(sig)
        X = np.abs(np.fft.rfft(sig * np.hanning(N), 2 * N))
        f = np.fft.rfftfreq(2 * N, 1 / 48000)
        S = 20 * np.log10(X + 1e-12)
        m = f < 3000
        ax.plot(f[m], S[m] - S[m].max(), lw=0.6, color=col, label=lab, alpha=0.85)
    for r in b.resonances[:6]:
        ax.axvline(r.freq, color="#1f77b4", lw=0.6, ls=":")
    ax.set_title(f"{ins.name}  (drone sim {b.drone:.1f} Hz, rec {f0:.1f} Hz)", fontsize=9)
    ax.set_ylim(-80, 3)
    ax.set_xlim(0, 3000)
axs[0, 0].legend(fontsize=8)
for ax in axs[-1]:
    ax.set_xlabel("Hz")
plt.tight_layout()
plt.savefig(FIG / "archive_neutral_spectra.png", dpi=80)

json.dump(out, open("doc/results/synth/archive_eval.json", "w"), indent=2, default=float)
print("done")
