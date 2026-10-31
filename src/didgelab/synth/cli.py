"""Command line interface.

    didgelab-synth info     GEOMETRY
    didgelab-synth synth    GEOMETRY [-s SCORE | -f SCOREFILE] [-o out.wav]
    didgelab-synth analyze  AUDIO --geometry GEOMETRY [--plot out.png] [--json out.json]
    didgelab-synth play     SONG.didge [--tempo BPM] [--loop N | --repeat N] [-o out.wav]
    didgelab-synth impedance AUDIO [--plot out.png] [--json out.json] [--compare GEOMETRY]

GEOMETRY is a JSON file with [[x_mm, d_mm], ...] (DidgeLab format) or the
name of an instrument of the Didge Archive (e.g. "Arusha").
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from .model import ModelParams

_DEFAULTS = ModelParams()


def load_geometry(spec, archive_path=None):
    p = Path(spec)
    if p.exists():
        return json.loads(p.read_text())
    from . import archive
    return archive.get(spec, archive_path).geometry


def _audio_formats():
    try:
        import soundfile as sf
        return set(sf.available_formats())
    except ImportError:          # without soundfile only wav (scipy)
        return {"WAV"}


def write_audio(path, y, fs):
    """Write audio with soundfile if installed (format from the file name, 16 bit
    for wav/flac), else as 16-bit wav with scipy; a non-wav name then gets the
    suffix .wav. Returns the path written."""
    path = Path(path)
    try:
        import soundfile as sf
    except ImportError:
        from scipy.io import wavfile
        path = path.with_suffix(".wav")
        wavfile.write(path, fs, (np.clip(y, -1, 1) * 32767).astype(np.int16))
        return path
    pcm = path.suffix.lower() in (".wav", ".flac", ".aiff", ".aif")
    sf.write(path, y, fs, subtype="PCM_16" if pcm else None)
    return path


def check_audio_out(out, *inputs):
    """Exit early (before rendering) if ``out`` is no audio file name or an input file."""
    ext = Path(out).suffix.lstrip(".").upper()
    if ext not in _audio_formats():
        sys.exit(f"-o {out}: not an audio file name (use e.g. .wav, .flac or .ogg)")
    for inp in inputs:
        if inp and Path(inp).exists() and Path(out).resolve() == Path(inp).resolve():
            sys.exit(f"-o {out}: would overwrite the input file")


def cmd_info(a):
    from .bore import Bore
    from didgelab.conv import freq_to_note_and_cent, note_name
    b = Bore(load_geometry(a.geometry, a.archive))
    print(f"length {b.geo.geo[-1][0]:.0f} mm, mouthpiece {b.geo.geo[0][1]:.1f} mm, bell {b.geo.geo[-1][1]:.1f} mm")
    print(f"{'#':>2} {'freq/Hz':>8} {'note':>6} {'cents':>6} {'|Z|/MPa':>8} {'Q':>5}")
    for i, r in enumerate(b.resonances[:8]):
        n, c = freq_to_note_and_cent(r.freq)
        print(f"{i:>2} {r.freq:8.1f} {note_name(n):>6} {c:+6.0f} {r.impedance / 1e6:8.2f} {r.q:5.1f}")


def cmd_synth(a):
    from . import synthesize, DEFAULT_SCORE
    from .model import ModelParams
    check_audio_out(a.out, a.geometry, a.file)
    score = a.score or (Path(a.file).read_text() if a.file else DEFAULT_SCORE)
    params = ModelParams(oq=a.oq, zs_mpa=a.zs, noise_db=a.noise)
    y, fs = synthesize(load_geometry(a.geometry, a.archive), score, fs=a.fs, params=params,
                       reverb=a.reverb, seed=a.seed)
    write_audio(a.out, y, fs)
    print(f"wrote {a.out} ({len(y) / fs:.1f} s)")


def cmd_analyze(a):
    from .analysis import load_audio
    from .bore import Bore
    from .detect import analyse, summarize
    from .transcribe import transcribe, plot_transcription
    b = Bore(load_geometry(a.geometry, a.archive))
    x, fs = load_audio(a.audio, 48000)
    if a.start or a.end:
        x = x[int(a.start * fs): int(a.end * fs) if a.end else None]
    fa, segs = transcribe(x, fs, b, model_path=a.model)
    for s in segs:
        info = ", ".join(f"{k}={v:.2f}" if isinstance(v, float) else f"{k}={v}" for k, v in s.info.items())
        print(f"{s.start:7.2f} {s.end:7.2f}  {s.label:16s} {info}")
    summ = summarize(fa, segs)
    print(json.dumps(summ, indent=2, default=float))
    if a.plot:
        plot_transcription(x, fs, fa, segs, a.plot, title=Path(a.audio).name)
    if a.json:
        Path(a.json).write_text(json.dumps({"summary": summ, "segments": [
            {"start": s.start, "end": s.end, "label": s.label, **s.info} for s in segs]}, indent=2, default=float))


def cmd_impedance(a):
    from .analysis import load_audio
    from .bore import Bore
    from . import impedance as imp
    x, fs = load_audio(a.audio, 48000)
    if a.start or a.end:
        x = x[int(a.start * fs): int(a.end * fs) if a.end else None]
    try:
        est = imp.estimate(x, fs, restarts=a.restarts, maxiter=a.maxiter,
                           progress=lambda r, c: print(f"restart {r + 1}/{a.restarts}: cost {c:.2f}", flush=True))
    except ValueError as e:
        sys.exit(f"{a.audio}: {e}")
    o = est.obs
    print(f"drone {o.f0:.1f} Hz, {o.voiced_s:.1f} s voiced, decay modes: "
          + (", ".join(f"{f:.0f} Hz x{n}" for f, n in o.mode_clusters(min_count=imp.Weights().decay_min_count))
             or "none"))
    true = Bore(load_geometry(a.compare, a.archive), fmax=3000) if a.compare else None
    print(f"{'#':>2} {'freq/Hz':>8} {'|Z|/MPa':>8} {'Q':>5}" + (f" {'true/Hz':>8} {'cents':>6}" if true else ""))
    for i, (f, z, q) in enumerate(est.resonances[:8]):
        line = f"{i:>2} {f:8.1f} {z:8.2f} {q:5.1f}"
        if true and i < len(true.resonances):
            ft = true.resonances[i].freq
            line += f" {ft:8.1f} {1200 * np.log2(f / ft):+6.0f}"
        print(line)
    if a.plot:
        imp.plot(est, a.plot, true_bore=true, title=Path(a.audio).name)
    if a.json:
        d = est.to_dict()
        d["impedance"] = {"freq_hz": est.freqs.tolist(), "abs_pa_s_m3": np.abs(est.zin).tolist(),
                          "band_db": est.band_db.tolist()}
        Path(a.json).write_text(json.dumps(d, indent=1, default=float))


def cmd_play(a):
    from . import song as songlib
    try:
        sng = songlib.load(a.song)
    except songlib.SongError as e:
        sys.exit(f"{a.song}: {e}")
    tempo = a.tempo or sng.tempo
    if a.list:
        print(songlib.describe(sng, tempo))
        return
    loop = a.loop is not None
    passes = a.loop if loop else a.repeat
    out = a.out or (Path(a.song).stem + ("-loop" if loop else "") + ".wav")
    check_audio_out(out, a.song)
    bore = songlib.load_bore(sng, a.instrument, a.archive)
    y = songlib.render(sng, bore, tempo=tempo, repeats=passes, loop=loop, fs=a.fs, seed=a.seed)
    write_audio(out, y, a.fs)
    kind = f"seamless loop of {passes} pass(es)" if loop else f"one shot, {passes} pass(es)"
    print(f"{sng.header.get('title', Path(a.song).stem)}: {sng.bars} bar(s) at {tempo:g} BPM, "
          f"{kind} -> {out} ({len(y) / a.fs:.2f} s)")
    if a.play:
        try:
            import sounddevice as sd
        except ImportError:
            sys.exit("--play needs the sounddevice package (pip install sounddevice)")
        sd.play(y, a.fs, loop=loop)
        try:
            sd.wait() if not loop else input("looping, press Enter to stop ")
        finally:
            sd.stop()


def main(argv=None):
    ap = argparse.ArgumentParser(prog="didgelab-synth", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--archive", default=None, help="path of a local Didge Archive clone")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("info", help="resonances of a bore (DidgeLab)")
    p.add_argument("geometry")
    p.set_defaults(func=cmd_info)

    p = sub.add_parser("synth", help="render a score")
    p.add_argument("geometry")
    p.add_argument("-s", "--score")
    p.add_argument("-f", "--file")
    p.add_argument("-o", "--out", default="didge.wav")
    p.add_argument("--fs", type=int, default=48000)
    p.add_argument("--oq", type=float, default=_DEFAULTS.oq, help="lip open quotient (brightness)")
    p.add_argument("--zs", type=float, default=_DEFAULTS.zs_mpa, help="lip source impedance, MPa s/m^3")
    p.add_argument("--noise", type=float, default=_DEFAULTS.noise_db, help="breath noise, dB")
    p.add_argument("--reverb", type=float, default=0.25)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=cmd_synth)

    p = sub.add_parser("analyze", help="detect playing techniques in a recording")
    p.add_argument("audio")
    p.add_argument("--geometry", required=True)
    p.add_argument("--model", default=None, help="trained detector (default: bundled)")
    p.add_argument("--start", type=float, default=0.0)
    p.add_argument("--end", type=float, default=0.0)
    p.add_argument("--plot")
    p.add_argument("--json")
    p.set_defaults(func=cmd_analyze)

    p = sub.add_parser("impedance", help="estimate the input impedance from a sound sample alone")
    p.add_argument("audio")
    p.add_argument("--start", type=float, default=0.0)
    p.add_argument("--end", type=float, default=0.0)
    p.add_argument("--restarts", type=int, default=3)
    p.add_argument("--maxiter", type=int, default=50)
    p.add_argument("--compare", help="geometry to compare with (JSON file or archive name)")
    p.add_argument("--plot")
    p.add_argument("--json")
    p.set_defaults(func=cmd_impedance)

    p = sub.add_parser("play", help="render a song file (.didge) as one shot or loop")
    p.add_argument("song")
    p.add_argument("-o", "--out", help="output wav (default: <song>.wav / <song>-loop.wav)")
    p.add_argument("-t", "--tempo", type=float, help="tempo in BPM (overrides the song)")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--loop", type=int, nargs="?", const=1, metavar="N",
                   help="seamless loop of N passes (default 1)")
    g.add_argument("--repeat", type=int, default=1, metavar="N",
                   help="one shot: play the song N times (default 1)")
    p.add_argument("--instrument", help="archive name or geometry JSON (overrides the song)")
    p.add_argument("--fs", type=int, default=48000)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--play", action="store_true", help="also play through the sound card (needs sounddevice)")
    p.add_argument("--list", action="store_true", help="only print the timed syllables")
    p.set_defaults(func=cmd_play)

    a = ap.parse_args(argv)
    a.func(a)


if __name__ == "__main__":
    main()
