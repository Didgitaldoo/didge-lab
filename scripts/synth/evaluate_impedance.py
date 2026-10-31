"""Evaluate the impedance estimation from sound samples on the Didge Archive.

    python scripts/synth/evaluate_impedance.py [--archive PATH] [--processes 4]

For every recording the geometry is hidden, Z_in is estimated from the
sound alone (``didgelab.synth.impedance``) and compared with DidgeLab's Z_in of
the measured geometry. Baseline: a cone whose first resonance is the drone.
Writes doc/results/synth/impedance_eval.json and doc/figures/impedance_*.png.

    python scripts/synth/evaluate_impedance.py --synthetic 12

instead renders performances on random bores (known ground truth) and
compares the fit with and without the mouth shape as a free parameter;
writes doc/results/synth/impedance_synthetic.json.
"""
import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

from didgelab.synth import archive
from didgelab.synth.analysis import load_audio
from didgelab.synth.bore import Bore
from didgelab.synth import impedance as imp

KINDS = ("neutral-sound", "song")
N_RES = 4


def cents(a, b):
    return 1200 * np.log2(np.asarray(a) / np.asarray(b))


def baseline_cone(f0):
    """31 -> 70 mm cone whose first impedance peak is f0."""
    def err(L):
        return Bore([[0, imp.MOUTHPIECE_MM], [L, 70.0]], fmax=400, df=0.25).resonances[0].freq - f0
    L = brentq(err, 700, 3000, xtol=0.5)
    return [[0, imp.MOUTHPIECE_MM], [L, 70.0]]


def resonance_errors(est_f, true_f):
    """Per true resonance: cents to the estimated resonance of the same index
    and to the nearest estimated resonance."""
    est_f = np.asarray(est_f)
    idx = [float(cents(est_f[i], t)) if i < len(est_f) else np.nan for i, t in enumerate(true_f)]
    near = [float(np.min(np.abs(cents(est_f, t)))) for t in true_f]
    return idx, near


def curve_error(geo_est, true_bore, fmin=50.0, fmax=1000.0):
    """rms difference (dB) of |Z_in| between fmin and fmax, common offset removed."""
    f = np.arange(fmin, fmax, 1.0)
    b = Bore(geo_est, fmax=fmax + 10)
    d = 20 * np.log10(np.abs(b.interp("Zin", f))) - 20 * np.log10(np.abs(true_bore.interp("Zin", f)))
    return float(np.sqrt(np.mean((d - d.mean()) ** 2)))


def job(args):
    name, kind, path, geometry, restarts, maxiter, weights, out = args
    tb = Bore(geometry, fmax=3000)
    true_f = [r.freq for r in tb.resonances[:N_RES]]
    x, fs = load_audio(path, 48000)
    est = imp.estimate(x, fs, weights=weights, restarts=restarts, maxiter=maxiter)
    est_f = [r[0] for r in est.resonances]
    idx, near = resonance_errors(est_f, true_f)
    row = {"instrument": name, "kind": kind, "true_hz": true_f, "est_hz": est_f[:N_RES + 2],
           "drone_hz": est.obs.f0, "n_decay_modes": len(est.obs.decays),
           "mode_clusters": est.obs.mode_clusters(min_count=weights.decay_min_count),
           "err_index_cents": idx, "err_nearest_cents": near,
           "curve_db": curve_error(est.geometry, tb), "terms": est.terms, "geometry": est.geometry}
    if np.isfinite(est.obs.f0):
        bg = baseline_cone(est.obs.f0)
        bf = [r.freq for r in Bore(bg, fmax=3000).resonances]
        bi, bn = resonance_errors(bf, true_f)
        row.update(base_err_index_cents=bi, base_err_nearest_cents=bn, base_curve_db=curve_error(bg, tb))
    if out:
        slug = name.lower().replace(" ", "-")
        imp.plot(est, f"{out}/impedance_{slug}_{kind}.png", true_bore=tb, title=f"{name} ({kind})")
    print(f"{name:26s} {kind:13s} idx {np.round(idx).tolist()} curve {row['curve_db']:.1f} dB", flush=True)
    return row


SYNTH_SCORE = "tempo = 90\ndoo:4 .:1 doo:2 .:1 di:.5 ga:.5 wa:1 oo:1 ee:1 .:1 doo:2 .:1"


def synth_job(args):
    from didgelab.synth import synthesize
    from didgelab.synth.learn import random_geometry
    i, fit_tract, restarts, maxiter = args
    geo = random_geometry(np.random.default_rng(100 + i))
    y, fs = synthesize(geo, SYNTH_SCORE, seed=i)
    tb = Bore(geo, fmax=3000)
    true_f = [r.freq for r in tb.resonances[:N_RES]]
    est = imp.estimate(y, fs, restarts=restarts, maxiter=maxiter, fit_tract=fit_tract)
    idx, near = resonance_errors([r[0] for r in est.resonances], true_f)
    print(f"synthetic {i:2d} tract={fit_tract!s:5s} idx {np.round(idx).tolist()}", flush=True)
    return {"bore": i, "fit_tract": fit_tract, "true_hz": true_f, "err_index_cents": idx,
            "err_nearest_cents": near, "harm_rms_db": float(np.sqrt(est.terms.get("harm", np.nan)) * 3.0),
            "curve_db": curve_error(est.geometry, tb), "n_mode_clusters": len(est.obs.mode_clusters(min_count=3))}


def summary(rows, key):
    E = np.abs(np.array([r[key] for r in rows if key in r], dtype=float))
    return {"median_cents_per_resonance": np.nanmedian(E, 0).round(1).tolist(),
            "within_50_cents_per_resonance": np.nanmean(E < 50, 0).round(2).tolist()}


def figure_archive(rows, archive_path=None, path="doc/figures/impedance_archive.png", fmax=800.0):
    """|Z_in| of the measured geometry vs. the estimates from the neutral drone and the song."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    insts = [i for i in archive.load(archive_path)]
    fig, axes = plt.subplots(3, 3, figsize=(12, 8.5), sharex=True)
    f = np.arange(30.0, fmax, 1.0)
    db = lambda geo: 20 * np.log10(np.abs(Bore(geo, fmax=fmax + 10).interp("Zin", f))) - 120
    for ax, ins in zip(axes.flat, insts):
        ax.plot(f, db(ins.geometry), color="k", lw=1.2, label="measured geometry")
        for kind, col in (("neutral-sound", "C0"), ("song", "C3")):
            r = next((r for r in rows if r["instrument"] == ins.name and r["kind"] == kind), None)
            if r is None:
                continue
            err = ", ".join(f"{e:+.0f}" for e in r["err_index_cents"][1:])
            ax.plot(f, db(r["geometry"]), color=col, lw=1, alpha=0.85,
                    label=f"{'drone' if kind == 'neutral-sound' else 'song'} ({err} c)")
            for fm, n in r["mode_clusters"]:
                if kind == "song" and fm < fmax:
                    ax.axvline(fm, color=col, ls=":", lw=0.8)
        ax.set_title(ins.name, fontsize=10)
        ax.legend(fontsize=7, loc="lower right")
        ax.set_ylim(-45, 30)
    for ax in axes[-1]:
        ax.set_xlabel("frequency / Hz")
    for ax in axes[:, 0]:
        ax.set_ylabel("|Z_in| / dB re 1 MPa s m$^{-3}$")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", default=None)
    ap.add_argument("--processes", type=int, default=4)
    ap.add_argument("--restarts", type=int, default=3)
    ap.add_argument("--maxiter", type=int, default=50)
    ap.add_argument("--only", default=None, help="comma-separated instrument names")
    ap.add_argument("--weights", default="", help="Weights overrides, e.g. smooth_mm=10,harm_db=4")
    ap.add_argument("--out", default="doc/results/synth/impedance_eval.json")
    ap.add_argument("--synthetic", type=int, default=0, help="evaluate on N random synthetic bores instead")
    ap.add_argument("--figures", default=None, help="directory for one plot per recording")
    ap.add_argument("--figure-only", action="store_true",
                    help="only redraw doc/figures/impedance_archive.png from the results JSON")
    a = ap.parse_args()
    if a.synthetic:
        jobs = [(i, ft, a.restarts, a.maxiter) for i in range(a.synthetic) for ft in (True, False)]
        with Pool(a.processes) as pool:
            rows = pool.map(synth_job, jobs)
        out = {"score": SYNTH_SCORE, "rows": rows}
        for ft in (True, False):
            rs = [r for r in rows if r["fit_tract"] == ft]
            key = "tract_fitted" if ft else "neutral_tract"
            out[key] = {"index": summary(rs, "err_index_cents"), "nearest": summary(rs, "err_nearest_cents"),
                        "harm_rms_db_median": float(np.median([r["harm_rms_db"] for r in rs])),
                        "curve_db_median": float(np.median([r["curve_db"] for r in rs]))}
            print(key, json.dumps(out[key]))
        Path("doc/results/synth/impedance_synthetic.json").write_text(json.dumps(out, indent=1, default=float))
        raise SystemExit
    if a.figure_only:
        figure_archive(json.loads(Path(a.out).read_text())["rows"], a.archive)
        raise SystemExit
    if a.figures:
        Path(a.figures).mkdir(parents=True, exist_ok=True)
    w = imp.Weights(**{k: type(getattr(imp.Weights, k))(v) for k, v in
                       (kv.split("=") for kv in a.weights.split(",") if kv)})
    jobs = []
    for ins in archive.load(a.archive):
        if a.only and ins.name not in a.only.split(","):
            continue
        for k in KINDS:
            if k in ins.samples:
                jobs.append((ins.name, k, ins.samples[k], ins.geometry, a.restarts, a.maxiter, w, a.figures))
    with Pool(a.processes) as pool:
        rows = pool.map(job, jobs)
    out = {"weights": vars(w), "rows": rows}
    for kind in KINDS + ("all",):
        rs = [r for r in rows if kind == "all" or r["kind"] == kind]
        out[kind] = {"estimate_index": summary(rs, "err_index_cents"),
                     "estimate_nearest": summary(rs, "err_nearest_cents"),
                     "baseline_index": summary(rs, "base_err_index_cents"),
                     "curve_db_median": float(np.median([r["curve_db"] for r in rs])),
                     "baseline_curve_db_median": float(np.median([r["base_curve_db"] for r in rs if "base_curve_db" in r]))}
        print(kind, json.dumps(out[kind]))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1, default=float))
    figure_archive(rows, a.archive)
