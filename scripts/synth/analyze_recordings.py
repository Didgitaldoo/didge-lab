"""Detect playing techniques in the long recordings of the Didge Archive.

    python scripts/synth/analyze_recordings.py [--archive PATH]

For every song / long recording: transcription plot (doc/figures),
segment list (doc/results/synth/transcriptions/*.json), a summary table
(doc/results/synth/recordings_summary.json and .md) and a copy-synthesis excerpt
(outputs/synth-audio/resynth-*.wav).
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import median_filter

from didgelab.synth.cli import write_audio
from didgelab.synth import archive
from didgelab.synth.analysis import load_audio
from didgelab.synth.bore import Bore
from didgelab.synth.detect import summarize
from didgelab.synth.transcribe import transcribe, plot_transcription, resynthesize

ap = argparse.ArgumentParser()
ap.add_argument("--archive", default=None)
ap.add_argument("--resynth-seconds", type=float, default=15.0)
args = ap.parse_args()

FIG = Path("doc/figures")
OUT = Path("doc/results/synth/transcriptions")
AUD = Path("outputs/synth-audio")
for p in (FIG, OUT, AUD):
    p.mkdir(parents=True, exist_ok=True)

root = archive.locate(args.archive)
jobs = []
for ins in archive.load(args.archive):
    if "song" in ins.samples:
        jobs.append((ins, ins.samples["song"]))
    if ins.name == "Open Didgeridoo" and (root / "audio-samples" / "open-didge-drags.wav").exists():
        jobs.append((ins, str(root / "audio-samples" / "open-didge-drags.wav")))


def tongue_moves(ti, dt, min_swing=0.35):
    """Number of front<->back excursions of the tongue index (hysteresis
    around the player's median position, which is the reference of the index)."""
    c = np.nanmedian(ti)
    x = median_filter(np.nan_to_num(ti, nan=c), 15)
    moves, state = 0, None
    lo, hi = c - min_swing / 2, c + min_swing / 2
    for v in x:
        s = "front" if v < lo else ("back" if v > hi else state)
        if state is not None and s != state:
            moves += 1
        state = s
    return moves


rows = []
for ins, path in jobs:
    name = Path(path).stem
    bore = Bore(ins.geometry)
    x, fs = load_audio(path, 48000)
    fa, segs = transcribe(x, fs, bore)
    summ = summarize(fa, segs)
    dur = summ["duration_s"]
    ev = summ["events"]
    ti = fa.tongue_index[np.isfinite(fa.tongue_index)]
    stops_f = ev.get("stop_front", 0)
    stops_b = ev.get("stop_back", 0)
    toots = {k: v for k, v in ev.items() if k.startswith("toot")}
    row = {
        "recording": name, "instrument": ins.name, "duration_s": round(dur, 1),
        "drone_hz": round(summ["drone_hz"], 1), "sim_drone_hz": round(bore.drone, 1),
        "toots": toots, "toot_s": round(sum(s.end - s.start for s in segs if s.label.startswith("toot")), 1),
        "stops_per_s": round((stops_f + stops_b) / dur, 2),
        "front_stop_share": round(stops_f / max(stops_f + stops_b, 1), 2),
        "trills": ev.get("trill", 0), "voice_s": round(sum(s.end - s.start for s in segs if s.label == "voice"), 1),
        "circular_breaths": ev.get("circular_breath", 0),
        "breath_interval_s": round(dur / ev["circular_breath"], 1) if ev.get("circular_breath") else None,
        "fricatives": ev.get("fricative", 0),
        "tongue_p10_p90": [round(float(v), 2) for v in np.percentile(ti, [10, 90])] if len(ti) else None,
        "tongue_coverage": round(float(np.mean(np.isfinite(fa.tongue_index[fa.mode >= 0]))), 2),
        "tongue_match": round(float(np.nanmedian(fa.tongue_match[fa.mode >= 0])), 2),
        "tongue_moves_per_s": round(tongue_moves(fa.tongue_index, fa.t[1] - fa.t[0]) / dur, 2),
    }
    rows.append(row)
    print(json.dumps(row), flush=True)
    plot_transcription(x, fs, fa, segs, FIG / f"transcription-{name}.png",
                       title=f"{name} ({ins.name})")
    json.dump({"summary": summ, "row": row,
               "segments": [{"start": round(s.start, 3), "end": round(s.end, 3), "label": s.label,
                             **{k: v for k, v in s.info.items()}} for s in segs]},
              open(OUT / f"{name}.json", "w"), indent=1, default=float)
    # copy synthesis of an excerpt
    n = int(min(args.resynth_seconds, dur) * fs)
    from didgelab.synth.detect import analyse
    fa_ex = analyse(x[:n], fs, bore, invert=False)
    y = resynthesize(fa_ex, bore)
    write_audio(AUD / f"resynth-{name}.mp3", y, fs)

json.dump(rows, open("doc/results/synth/recordings_summary.json", "w"), indent=1)

# markdown table for the article
hdr = ["recording", "instrument", "dur s", "drone Hz (sim)", "toots", "stops/s", "front stops",
       "trills", "voice s", "circ. breaths", "tongue p10-p90", "tongue coverage", "tongue match", "tongue moves/s"]
lines = ["| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
for r in rows:
    toots = ", ".join(f"{k}:{v}" for k, v in sorted(r["toots"].items())) or "-"
    lines.append("| " + " | ".join(str(v) for v in [
        r["recording"], r["instrument"], r["duration_s"], f"{r['drone_hz']} ({r['sim_drone_hz']})", toots,
        r["stops_per_s"], r["front_stop_share"], r["trills"], r["voice_s"], r["circular_breaths"],
        "-".join(map(str, r["tongue_p10_p90"] or [])), r["tongue_coverage"], r["tongue_match"],
        r["tongue_moves_per_s"]]) + " |")
Path("doc/results/synth/recordings_summary.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
