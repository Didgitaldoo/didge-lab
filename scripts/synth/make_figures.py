"""Figures and audio examples for the article.

    python scripts/synth/make_figures.py [--archive PATH]
"""
import argparse
from pathlib import Path

import numpy as np
import soundfile as sf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from didgelab.synth import archive, Bore, ModelParams, VOWELS, VocalTract, synthesize, DEFAULT_SCORE
from didgelab.synth import vocaltract
from didgelab.synth.analysis import load_audio, stft
from didgelab.synth.excitation import hindle_waveform, lips_waveform, hindle_harmonics, lips_harmonics
from didgelab.synth.model import filter_gain, source
from didgelab.synth.synth import Synthesizer, Controls

ap = argparse.ArgumentParser()
ap.add_argument("--archive", default=None)
args = ap.parse_args()
FIG = Path("doc/figures")
AUD = Path("outputs/synth-audio")
FIG.mkdir(parents=True, exist_ok=True)
AUD.mkdir(parents=True, exist_ok=True)
(AUD / "songs").mkdir(exist_ok=True)
plt.rcParams.update({"font.size": 9})

arusha = archive.get("Arusha", args.archive)
matema = archive.get("Matema", args.archive)

# ---------------------------------------------------------------- 1 bore
b = Bore(arusha.geometry)
f = np.arange(20, 3000, 0.5)
g = np.array(arusha.geometry)
fig = plt.figure(figsize=(10, 7.2))
gs = fig.add_gridspec(3, 1, height_ratios=[0.9, 2, 2])
ax0 = fig.add_subplot(gs[0])
ax0.fill_between(g[:, 0], g[:, 1] / 2, -g[:, 1] / 2, color="#c8a165", lw=0)
ax0.plot(g[:, 0], g[:, 1] / 2, color="#7a5a2b", lw=0.8)
ax0.plot(g[:, 0], -g[:, 1] / 2, color="#7a5a2b", lw=0.8)
ax0.set_aspect("equal", adjustable="datalim")      # same scale on both axes
ax0.set_xlim(-20, g[-1, 0] + 20)
ax0.set_title("Arusha bore (Didge Archive), true proportions; mouthpiece left, bell right")
ax0.set_xlabel("x / mm")
ax0.set_ylabel("r / mm")
ax1 = fig.add_subplot(gs[1])
ax1.semilogy(f, np.abs(b.interp("Zin", f)) / 1e6, lw=0.8, label="|Z_in| (DidgeLab)")
for r in b.resonances[:7]:
    ax1.axvline(r.freq, color="gray", lw=0.4, ls=":")
ax1.set_ylabel("MPa s / m^3")
ax1.legend()
ax2 = fig.add_subplot(gs[2], sharex=ax1)
ax2.semilogy(f, np.abs(b.interp("H", f)), lw=0.8, color="#d62728", label="|H| = |U_bell / U_mouth|")
ax2.set_xlabel("frequency / Hz")
ax2.legend()
plt.tight_layout()
plt.savefig(FIG / "bore_arusha.png", dpi=90)
plt.close()

# ---------------------------------------------------------------- 1b block diagrams
from matplotlib.patches import FancyBboxPatch


def box(ax, x, y, w, h, text, color):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=color, ec="#333333", lw=1.0))
    ax.text(x, y, text, ha="center", va="center", fontsize=9)


def arrow(ax, x0, y0, x1, y1, text=""):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color="#333333", lw=1.2, shrinkA=2, shrinkB=2))
    if text:
        ax.text((x0 + x1) / 2, (y0 + y1) / 2 + 0.12, text, ha="center", va="bottom", fontsize=7.5,
                color="#444444")


fig, ax = plt.subplots(figsize=(11, 4.2))
ax.set_xlim(0, 11)
ax.set_ylim(0, 4.2)
ax.axis("off")
C_BORE, C_PLAY, C_CORE, C_OUT = "#f3e3c3", "#d9e8f5", "#e6dcf2", "#dff0d8"
box(ax, 1.1, 3.4, 1.9, 0.75, "bore geometry\n[x_mm, d_mm]", C_BORE)
box(ax, 3.6, 3.4, 2.1, 0.75, "DidgeLab\ntransmission line", C_BORE)
box(ax, 6.4, 3.4, 2.6, 0.75, "Z_in(f), H(f),\nresonances, group delay", C_BORE)
box(ax, 1.1, 1.3, 1.9, 0.75, "score / song\n(.didge)", C_PLAY)
box(ax, 3.6, 1.3, 2.1, 0.75, "player:\nplaying techniques", C_PLAY)
box(ax, 6.4, 1.3, 2.6, 0.9, "control signals (500 Hz)\nf0, amplitude, lip shape,\ntongue, jaw, glottis, voice", C_PLAY)
box(ax, 6.4, 2.35, 2.2, 0.55, "vocal tract Z_vt(f)", C_PLAY)
box(ax, 9.3, 2.35, 2.5, 1.0, "coupling\nG = jf H Zs / (Zs+Zvt+Zin)\n+ bore inertia", C_CORE)
box(ax, 9.3, 0.6, 2.5, 0.75, "partial bank + noise\n+ room  ->  audio", C_OUT)
arrow(ax, 2.05, 3.4, 2.55, 3.4)
arrow(ax, 4.65, 3.4, 5.1, 3.4)
arrow(ax, 2.05, 1.3, 2.55, 1.3)
arrow(ax, 4.65, 1.3, 5.1, 1.3)
arrow(ax, 6.4, 1.75, 6.4, 2.08)
arrow(ax, 7.7, 3.4, 8.6, 2.85)
arrow(ax, 7.5, 2.35, 8.05, 2.35)
arrow(ax, 7.7, 1.3, 8.4, 1.85)
ax.text(8.25, 1.2, "f0, lip shape,\namplitude", fontsize=7.5, color="#444444")
arrow(ax, 9.3, 1.85, 9.3, 0.98)
plt.tight_layout()
plt.savefig(FIG / "pipeline.png", dpi=110)
plt.close()

fig, ax = plt.subplots(figsize=(11, 2.8))
ax.set_xlim(0, 11)
ax.set_ylim(0, 2.8)
ax.axis("off")
box(ax, 0.8, 1.5, 1.2, 0.8, "lungs", "#eeeeee")
box(ax, 2.9, 1.5, 2.2, 0.9, "vocal tract\nZ_vt (tongue, jaw,\nglottis)", C_PLAY)
box(ax, 5.4, 1.5, 2.0, 0.9, "lips\nflow source U_s\nsource impedance Z_s", "#f6d5d5")
box(ax, 7.8, 1.5, 2.0, 0.9, "bore\nZ_in, H", C_BORE)
box(ax, 10.0, 1.5, 1.4, 0.8, "bell\nradiation", C_OUT)
for x0, x1 in [(1.4, 1.8), (4.0, 4.4), (6.4, 6.8), (8.8, 9.3)]:
    arrow(ax, x0, 1.5, x1, 1.5)
ax.text(5.4, 0.45, "the same flow U passes vocal tract, lips and bore:   "
        "U = U_s Z_s / (Z_s + Z_vt + Z_in)", ha="center", fontsize=9.5)
ax.text(9.05, 1.62, "U_bell = H U", ha="center", va="bottom", fontsize=8, color="#444444")
plt.tight_layout()
plt.savefig(FIG / "coupling.png", dpi=110)
plt.close()

# ---------------------------------------------------------------- 2 lip source
fig, axs = plt.subplots(1, 2, figsize=(10, 3.4))
th = np.arange(4096) / 4096
axs[0].plot(th, hindle_waveform(0.85) / np.ptp(hindle_waveform(0.85)), label="Hindle & Posnett (a/a0 = 0.85)")
axs[0].plot(th, lips_waveform(0.45, 0.85) / np.ptp(lips_waveform(0.45, 0.85)), label="beating lips (OQ 0.45, skew 0.85)")
axs[0].set_xlabel("time / period")
axs[0].set_ylabel("flow (normalised)")
axs[0].legend(fontsize=7)
n = np.arange(1, 65)
axs[1].plot(n, 20 * np.log10(np.abs(hindle_harmonics(0.85, 64))), "o-", ms=2, label="Hindle")
axs[1].plot(n, 20 * np.log10(np.abs(lips_harmonics(0.45, 0.85, 64))), "o-", ms=2, label="beating lips")
axs[1].set_xscale("log")
axs[1].set_ylim(-120, 5)
axs[1].set_xlabel("harmonic number")
axs[1].set_ylabel("dB")
axs[1].legend()
plt.tight_layout()
plt.savefig(FIG / "lip_source.png", dpi=90)
plt.close()

# ---------------------------------------------------------------- 3 vocal tract
fig, axs = plt.subplots(1, 2, figsize=(11, 3.8))
x = (np.arange(vocaltract.N_SECTIONS) + 0.5) / vocaltract.N_SECTIONS * vocaltract.VT_LENGTH * 100
ff = np.arange(50, 4500, 5.0)
for k in ["neutral", "i", "a", "u"]:
    axs[0].step(x, vocaltract.area_function(VOWELS[k]) * 1e4, where="mid", label=k)
    axs[1].semilogy(ff, np.abs(vocaltract.impedance(VOWELS[k], ff)) / 1e6, lw=0.9, label=k)
closed = VOWELS["i"].with_(tongue_pos=0.05, tongue_height=1.0)
axs[0].step(x, vocaltract.area_function(closed) * 1e4, where="mid", color="k", ls="--", label="t closure")
axs[0].set_xlabel("distance from the lips / cm")
axs[0].set_ylabel("area / cm^2")
axs[0].legend(fontsize=7)
axs[1].set_xlabel("frequency / Hz")
axs[1].set_ylabel("|Z_vt| / MPa s m^-3")
axs[1].legend(fontsize=7)
plt.tight_layout()
plt.savefig(FIG / "vocal_tract.png", dpi=90)
plt.close()

# ---------------------------------------------------------------- 4 tongue sweep: model vs Matema
bm = Bore(matema.geometry)
p = ModelParams()
c = Controls.constant(9.0, 500, bm.drone, p, VOWELS["neutral"])
t = np.arange(c.n) / c.rate
c.vt[:, 0] = 0.4 + 0.55 * (0.5 - 0.5 * np.cos(2 * np.pi * t / 3.0))
c.vt[:, 1] = 0.5
c.vt[:, 2] = 0.45
c.vt[:, 3] = 0.3
y = Synthesizer(bm, p, seed=0).render(c, reverb=0.15)
sf.write(AUD / "tongue-sweep-matema.mp3", y, 48000, format="MP3")
xr, fs = load_audio(matema.samples["song"], 48000)
xr = xr[3 * fs:12 * fs]
fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
for ax, sig, title in [(axs[0], xr, "recording: Matema (Didge Archive), 3-12 s"),
                       (axs[1], y, "synthesis: Matema bore, tongue position swept front -> back -> front")]:
    tt, fr, X = stft(sig, 48000, n_fft=4096, win=2048, hop=480)
    S = 20 * np.log10(X + 1e-9)
    S -= S.max()
    m = fr < 4500
    ax.pcolormesh(tt, fr[m], S[:, m].T, vmin=-75, vmax=0, shading="auto", cmap="magma")
    ax.set_title(title)
    ax.set_ylabel("Hz")
axs[1].set_xlabel("time / s")
plt.tight_layout()
plt.savefig(FIG / "tongue_sweep.png", dpi=80)
plt.close()

# ---------------------------------------------------------------- 4b zoom into articulations
tamaki1 = archive.get("Tamaki1", args.archive)
xr, fs = load_audio(tamaki1.samples["song"], 48000)
xr = xr[10 * fs:13 * fs]
tt, fr, X = stft(xr, fs, n_fft=4096, win=1024, hop=240)
S = 20 * np.log10(X + 1e-9)
S -= S.max()
fig, axs = plt.subplots(3, 1, figsize=(10, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1.2, 1]})
m = fr < 5000
axs[0].pcolormesh(tt, fr[m], S[:, m].T, vmin=-75, vmax=0, cmap="magma", shading="auto")
axs[0].set_ylabel("Hz")
axs[0].set_title("Tamaki1 (Didge Archive), 10-13 s")
P = X ** 2
for lo, hi, col in [(0, 300, "k"), (300, 1000, "b"), (1000, 2500, "g"), (2500, 5000, "r")]:
    axs[1].plot(tt, 10 * np.log10(P[:, (fr >= lo) & (fr < hi)].sum(1) + 1e-12), col, lw=0.8, label=f"{lo}-{hi} Hz")
axs[1].legend(fontsize=7, ncol=4)
axs[1].set_ylabel("band dB")
axs[2].plot(tt, 10 * np.log10(P.sum(1)), "k", lw=0.8)
axs[2].set_ylabel("total dB")
axs[2].set_xlabel("time / s")
plt.tight_layout()
plt.savefig(FIG / "zoom_tamaki1.png", dpi=80)
plt.close()

# ---------------------------------------------------------------- 5 coupling illustration
fig, ax = plt.subplots(figsize=(9, 3.6))
ff = np.arange(50, 4000, 2.0)
zs = p.zs_mpa * 1e6
for k, col in [("i", "#1f77b4"), ("u", "#d62728")]:
    zvt = vocaltract.impedance(VOWELS[k], ff)
    ax.semilogy(ff, np.abs(zvt) / 1e6, color=col, lw=0.9, label=f"|Z_vt| '{k}'")
ax.semilogy(ff, np.abs(bm.interp("Zin", ff)) / 1e6, color="gray", lw=0.5, label="|Z_in| bore")
ax.axhline(p.zs_mpa, color="k", ls="--", lw=0.8, label="Z_s (lips)")
ax.set_xlabel("frequency / Hz")
ax.set_ylabel("MPa s m^-3")
ax.legend(fontsize=7, ncol=2)
plt.tight_layout()
plt.savefig(FIG / "impedances.png", dpi=90)
plt.close()

# ---------------------------------------------------------------- 6 song example
from didgelab.synth import song as songlib
SONGS = Path("src/didgelab/synth/songs")
fig, axs = plt.subplots(2, 2, figsize=(12, 6), gridspec_kw={"height_ratios": [1, 2.3]}, sharex="col")
for col, name in enumerate(["drone-back-to-front", "dong-tiki-ritititi"]):
    sg = songlib.load(SONGS / f"{name}.didge")
    bo = songlib.load_bore(sg, archive_path=args.archive)
    y = songlib.render(sg, bo)
    sf.write(AUD / "songs" / f"{name}.mp3", y, 48000, format="MP3")
    from didgelab.synth.techniques import Player
    pl = Player(bo, seed=0)
    c = pl.controls(songlib.events(sg), {"humanize": 0}, tail=0.0, tongue=songlib.tongue_track(sg))
    tc = np.arange(c.n) / c.rate
    axs[0, col].plot(tc, c.vt[:, 0], color="#1f77b4", lw=1.0)
    axs[0, col].set_ylim(1.05, -0.25)
    axs[0, col].set_ylabel("tongue position\n0 tip .. 1 back", fontsize=8)
    for e in songlib.events(sg):
        if e.kind == "syllable" and e.consonant:
            axs[0, col].text(e.start, -0.12, e.label.replace("accent+", "").replace("+", ""), fontsize=7)
    axs[0, col].set_title(f"{name}.didge", fontsize=10)
    tt, fr, X = stft(y, 48000, n_fft=4096, win=1024, hop=240)
    S = 20 * np.log10(X + 1e-9)
    S -= S.max()
    m = fr < 4500
    axs[1, col].pcolormesh(tt, fr[m], S[:, m].T, vmin=-75, vmax=0, shading="auto", cmap="magma")
    axs[1, col].set_xlabel("time / s")
    axs[1, col].set_ylabel("Hz")
plt.tight_layout()
plt.savefig(FIG / "songs.png", dpi=80)
plt.close()

# ---------------------------------------------------------------- audio examples
showcase = {
    "drone": "tempo = 60\nuh:4",
    "tongue-front-back": "tempo = 80\nee:1 oo:1 ee:1 oo:1 ee:.5 oo:.5 ee:.5 oo:.5 ee:2",
    "stops-diga": "tempo = 110\ndi:.5 ga:.5 di:.5 ga:.5 ti:.5 ka:.5 ti:.5 ka:.5 da:.25 da:.25 ga:.5 da:.25 da:.25 ga:.5 doo:2",
    "trill": "tempo = 90\ndoo:1 rr:2 doo:1 d:.25 rr:.75 doo:1",
    "toots": "tempo = 90\ndoo:1 *too:1 doo:1 *ta:.5 *ta:.5 doo:1 **too:1 doo:1",
    "vocalisation": "tempo = 80\ndoo:1 ~oo:2@C4 doo:1 ~a:2@G3 doo:1",
    "circular-breathing": "tempo = 100\ndi:.5 ga:.5 di:.5 ga:.5 ^ di:.5 ga:.5 di:.5 ga:.5 ^ di:.5 ga:.5 doo:2",
    "groove": DEFAULT_SCORE,
}
for name, score in showcase.items():
    y, fs = synthesize(arusha.geometry, score, seed=1)
    sf.write(AUD / f"arusha-{name}.mp3", y, fs, format="MP3")
for inst in ["Malveira", "Kizimkazi", "Tamaki1", "Matema"]:
    y, fs = synthesize(archive.get(inst, args.archive).geometry, DEFAULT_SCORE, seed=2)
    sf.write(AUD / f"{inst.lower()}-groove.mp3", y, fs, format="MP3")
# simple straight cone, the Didge2Wav example bore
y, fs = synthesize([[0, 32], [1500, 43]], DEFAULT_SCORE, seed=3)
sf.write(AUD / "didge2wav-cone-groove.mp3", y, fs, format="MP3")
print("done")
