"""
Songs: a text format for didgeridoo grooves and loops (``.didge`` files).

A song has a header and lanes, written in bars::

    title:      Dong-ti
    instrument: Arusha        # Didge Archive name or path to a geometry JSON
    tempo:      100           # quarter notes per minute
    meter:      4/4

    [rhythm]
    | dong-ti dong-ti dong-ti dong-ti |

    [tongue]
    | oo ~ ~ ee |

**Timing.** Every bar is divided equally among its tokens, unless a token
carries an explicit length (``:1/8`` = an eighth note, ``:3/16``, ``:2`` =
two beats); the explicit lengths are taken first and the remaining tokens
share the rest of the bar.

**Rhythm lane.** A token is a word made of syllables. Hyphens split the
token's time equally (``dong-ti``: dong and ti get half each); the syllables
inside a part share that part equally (``tiki``: ti and ki), so
``dong-tiki`` is an eighth "dong" and two sixteenths "ti" "ki".
A syllable is ``[consonant] vowel [ending]``:

* consonants: t d k g p (stops), r (tongue flap), rr (rolled r),
  y w (glides), h ch s (fricatives)
* vowels: ee i e ae a o oo u uh (uh = relaxed drone)
* endings: ng (dong), n

Other tokens: ``.`` rest, ``_`` hold the previous syllable for this slot,
``^`` circular breathing. Prefixes: ``!`` accent, ``*`` toot (``**`` second
toot), ``~`` vocalise (pitch with ``@C4``, default from ``voice:``).

**Tongue lane** (optional). Overrides the mouth shape given by the
syllables:

* a value (``ee e ae a o oo``, ``front`` ``mid`` ``back`` or a number from
  0 = front to 1 = back) holds that tongue position during its slot,
* ``.`` keeps holding the previous value (the next value then follows
  quickly),
* ``~`` glides: the tongue moves linearly from the end of the previous value
  to the start of the next value,
* ``auto`` (or ``*``) leaves the mouth to the syllables.

So ``| oo ~ ~ ee |`` holds "oo" for a beat, glides for two beats and reaches
"ee" on beat four, while ``| oo . . ee |`` jumps on beat four.
"""

import re
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path

import numpy as np

from .techniques import Event, Player, note_to_freq, CONS_ORDER
from .vocaltract import VOWELS

VOWEL_MAP = {"ee": "i", "i": "i", "e": "e", "ae": "ae", "ai": "e", "a": "a", "aa": "a",
             "o": "o", "oh": "o", "ow": "o", "oo": "u", "u": "u", "ou": "u", "uh": "neutral"}
VOWEL_TOKENS = sorted(VOWEL_MAP, key=len, reverse=True)
CODAS = ["ng", "n"]
ONSETS = sorted(CONS_ORDER, key=len, reverse=True)

HEADER_KEYS = {"title", "instrument", "tempo", "meter", "swing", "humanize", "glide", "voice",
               "reverb", "brightness", "breath", "seed"}


class SongError(ValueError):
    pass


@dataclass
class Syllable:
    consonant: str
    vowel: str
    coda: str = ""


@dataclass
class Song:
    header: dict
    rhythm: list = field(default_factory=list)   # list of (start_beats, dur_beats, item)
    tongue: list = field(default_factory=list)   # list of (start, dur, token)
    bars: int = 0

    @property
    def beats_per_bar(self):
        num, den = self.meter
        return num * 4 / den

    @property
    def meter(self):
        num, den = str(self.header.get("meter", "4/4")).split("/")
        return int(num), int(den)

    @property
    def length_beats(self):
        return self.bars * self.beats_per_bar

    @property
    def tempo(self):
        return float(self.header.get("tempo", 100))


# ---------------------------------------------------------------- parsing
def syllabify(part):
    """Split a word part such as "tiki", "dong", "ritititi" into syllables."""
    out, i = [], 0
    while i < len(part):
        cons = next((c for c in ONSETS if part.startswith(c, i)), "")
        i += len(cons)
        vow = next((v for v in VOWEL_TOKENS if part.startswith(v, i)), "")
        i += len(vow)
        if not cons and not vow:
            raise SongError(f"cannot read {part[i:]!r} in {part!r}")
        coda = ""
        for cd in CODAS:
            rest = part[i + len(cd):]
            if vow and part.startswith(cd, i) and (rest == "" or not any(rest.startswith(v) for v in VOWEL_TOKENS)):
                coda = cd
                i += len(cd)
                break
        out.append(Syllable(cons, VOWEL_MAP[vow] if vow else "neutral", coda))
    return out


def _duration(spec):
    """':1/8' -> 0.5 beats (note value), ':2' -> 2 beats."""
    if "/" in spec:
        return float(Fraction(spec)) * 4
    return float(spec)


def _split_bars(text):
    text = " ".join(line.split("#", 1)[0] for line in text.splitlines())
    bars = [b.split() for b in text.split("|")]
    return [b for b in bars if b]


def _lay_out(bars, beats_per_bar, lane):
    """Assign start and length (in beats) to every token of every bar."""
    out = []
    for k, bar in enumerate(bars):
        toks, durs = [], []
        for tok in bar:
            if ":" in tok and not tok.startswith(":"):
                name, spec = tok.rsplit(":", 1)
                try:
                    durs.append(_duration(spec))
                except (ValueError, ZeroDivisionError):
                    raise SongError(f"{lane} bar {k + 1}: bad length in {tok!r}")
                toks.append(name)
            else:
                toks.append(tok)
                durs.append(None)
        fixed = sum(d for d in durs if d is not None)
        free = sum(d is None for d in durs)
        rest = beats_per_bar - fixed
        if rest < -1e-9 or (free == 0 and abs(rest) > 1e-6):
            raise SongError(f"{lane} bar {k + 1}: lengths add up to {fixed:g} beats, "
                            f"the bar has {beats_per_bar:g}")
        if free and rest <= 1e-9:
            raise SongError(f"{lane} bar {k + 1}: no time left for {free} token(s)")
        t = k * beats_per_bar
        for tok, d in zip(toks, durs):
            d = d if d is not None else rest / free
            out.append((t, d, tok))
            t += d
    return out


RHYTHM_RE = re.compile(r"^(?P<mods>[!*~]*)(?P<word>[a-z\-]+)(?:@(?P<note>[A-Ga-g][#b]?-?\d))?$")


def parse(text, path=None):
    header, lanes, cur = {}, {}, None
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        m = re.match(r"^\s*\[(\w+)\]\s*$", line)
        if m:
            cur = m.group(1).lower()
            if cur not in ("rhythm", "tongue"):
                raise SongError(f"unknown lane [{cur}] (use [rhythm] and [tongue])")
            lanes[cur] = ""
            continue
        m = re.match(r"^\s*(\w+)\s*:\s*(.+)$", line)
        if cur is None and m:
            key = m.group(1).lower()
            if key not in HEADER_KEYS:
                raise SongError(f"unknown header field {key!r}")
            header[key] = m.group(2).strip()
            continue
        if cur is None:
            raise SongError(f"text outside a lane: {line.strip()!r}")
        lanes[cur] += " " + line
    if "rhythm" not in lanes:
        raise SongError("a song needs a [rhythm] lane")
    if path is not None:
        header["_dir"] = str(Path(path).parent)
    song = Song(header)
    bpb = song.beats_per_bar
    rb = _split_bars(lanes["rhythm"])
    tb = _split_bars(lanes.get("tongue", ""))
    song.bars = max(len(rb), len(tb))
    song.rhythm = _lay_out(rb, bpb, "rhythm")
    song.tongue = _lay_out(tb, bpb, "tongue")
    _check_rhythm(song)
    _check_tongue(song)
    return song


def load(path):
    return parse(Path(path).read_text(), path)


def _check_rhythm(song):
    for _, _, tok in song.rhythm:
        if tok in (".", "_", "^"):
            continue
        m = RHYTHM_RE.match(tok)
        if not m:
            raise SongError(f"cannot read rhythm token {tok!r}")
        for part in m.group("word").split("-"):
            if part:
                syllabify(part)


TONGUE_WORDS = {"front": 0.0, "mid": 0.5, "back": 1.0}


def tongue_value(tok):
    """Vocal-tract vector of a tongue token; None for 'auto'."""
    if tok in ("auto", "*"):
        return None
    if tok in VOWEL_MAP:
        return VOWELS[VOWEL_MAP[tok]].as_array()
    if tok in TONGUE_WORDS:
        x = TONGUE_WORDS[tok]
    else:
        try:
            x = float(tok)
        except ValueError:
            raise SongError(f"unknown tongue position {tok!r}")
        if not 0 <= x <= 1:
            raise SongError(f"tongue position {tok!r} outside 0 (front) .. 1 (back)")
    return tongue_axis(x)


def tongue_axis(x):
    """Articulated mouth along the tongue axis: 0 = front ("ee") .. 1 = back ("oo")."""
    return np.array([0.4 + 0.55 * x, 0.5, 0.45, 0.3])


def _check_tongue(song):
    for _, _, tok in song.tongue:
        if tok not in (".", "~", "_"):
            tongue_value(tok)


# ---------------------------------------------------------------- timing
def swing_warp(beats, swing):
    """Swing: the second eighth of each beat moves from 50 % to ``swing`` % of the beat."""
    if not swing or abs(swing - 0.5) < 1e-9:
        return np.asarray(beats, dtype=float)
    b = np.asarray(beats, dtype=float)
    k = np.floor(b)
    x = b - k
    y = np.where(x < 0.5, x * swing / 0.5, swing + (x - 0.5) * (1 - swing) / 0.5)
    return k + y


def _swing(header):
    s = header.get("swing")
    if s is None:
        return 0.0
    s = float(str(s).rstrip("%"))
    return s / 100 if s > 1 else s


# ---------------------------------------------------------------- events
def events(song, tempo=None, repeats=1, offset_beats=0.0):
    """Player events (in seconds) for ``repeats`` passes through the song."""
    tempo = tempo or song.tempo
    spb = 60.0 / tempo
    sw = _swing(song.header)
    default_voice = song.header.get("voice", "A3")
    out = []

    def sec(b):
        return float(swing_warp(b, sw)) * spb

    for r in range(repeats):
        base = offset_beats + r * song.length_beats
        last = None
        for start, dur, tok in song.rhythm:
            b0, b1 = base + start, base + start + dur
            if tok == "_":
                if last is not None:
                    last[-1].dur = sec(b1) - last[-1].start
                continue
            if tok == ".":
                out.append(Event(sec(b0), sec(b1) - sec(b0), kind="rest"))
                last = None
                continue
            if tok == "^":
                out.append(Event(sec(b0), sec(b1) - sec(b0), kind="breath"))
                last = None
                continue
            m = RHYTHM_RE.match(tok)
            mods = m.group("mods")
            voice = note_to_freq(m.group("note") or default_voice) if "~" in mods else 0.0
            toot = mods.count("*")
            parts = [p for p in m.group("word").split("-") if p]
            evs = []
            for i, part in enumerate(parts):
                p0 = b0 + (b1 - b0) * i / len(parts)
                p1 = b0 + (b1 - b0) * (i + 1) / len(parts)
                syl = syllabify(part)
                for j, s in enumerate(syl):
                    s0 = p0 + (p1 - p0) * j / len(syl)
                    s1 = p0 + (p1 - p0) * (j + 1) / len(syl)
                    evs.append(Event(sec(s0), sec(s1) - sec(s0), s.consonant, s.vowel,
                                     accent=("!" in mods and not evs), toot=toot, voice=voice,
                                     coda=s.coda))
            out += evs
            last = evs
    return out


def tongue_track(song, tempo=None, repeats=1):
    """Callable t (s) -> (n, 4) vocal-tract targets (NaN rows where 'auto')."""
    if not song.tongue:
        return None
    tempo = tempo or song.tempo
    spb = 60.0 / tempo
    sw = _swing(song.header)
    xs, vs, auto = [], [], []
    for r in range(repeats):
        base = r * song.length_beats
        prev = None
        for start, dur, tok in song.tongue:
            b0, b1 = base + start, base + start + dur
            if tok == "~":
                continue
            if tok in (".", "_"):
                if prev is not None:
                    xs.append(b1)
                    vs.append(prev)
                continue
            v = tongue_value(tok)
            if v is None:
                auto.append((b0, b1))
                prev = None
                continue
            xs += [b0, b1]
            vs += [v, v]
            prev = v
        # after the lane ends (shorter than the rhythm): back to the syllables
        lane_end = base + song.tongue[-1][0] + song.tongue[-1][1]
        if lane_end < base + song.length_beats - 1e-9:
            auto.append((lane_end, base + song.length_beats))
    if not xs:
        return lambda t: np.full((len(t), 4), np.nan)
    xs = swing_warp(np.array(xs), sw) * spb
    vs = np.array(vs)
    auto = [(float(swing_warp(a, sw)) * spb, float(swing_warp(b, sw)) * spb) for a, b in auto]

    def track(t):
        t = np.asarray(t, dtype=float)
        out = np.stack([np.interp(t, xs, vs[:, k]) for k in range(4)], 1)
        for a, b in auto:
            out[(t >= a) & (t < b)] = np.nan
        return out

    return track


# ---------------------------------------------------------------- rendering
def render(song, bore, tempo=None, repeats=1, loop=False, fs=48000, seed=None, params=None,
           crossfade=0.03, fadeout=0.05):
    """Render a song. Returns the audio.

    ``loop=False``: the song is played ``repeats`` times, with a start and a
    decay (one shot). ``loop=True``: a seamless loop of ``repeats`` passes is
    returned; it is cut from a longer performance (one pass before and after,
    so the sound is already running at the loop start) and its end is
    crossfaded into its start, so the file can be looped without a click.
    The last ``fadeout`` seconds of the loop are faded out, so the file also
    ends without a click when played once.
    """
    from .model import ModelParams
    from .synth import Synthesizer

    h = song.header
    tempo = float(tempo or song.tempo)
    params = params or ModelParams()
    if "brightness" in h:      # 0 dull .. 1 bright -> open quotient 0.7 .. 0.25
        params = params.with_(oq=0.7 - 0.45 * float(h["brightness"]))
    if "breath" in h:
        params = params.with_(noise_db=float(h["breath"]))
    seed = int(h.get("seed", 0)) if seed is None else seed
    player = Player(bore, params, seed=seed, glide=float(h.get("glide", 0.05)))
    settings = {"humanize": float(h.get("humanize", 1.0)), "voice": h.get("voice", "A3")}
    reverb = float(h.get("reverb", 0.25))
    cycle = song.length_beats * 60.0 / tempo
    syn = Synthesizer(bore, params, fs=fs, seed=seed)

    if not loop:
        evs = events(song, tempo, repeats)
        # the lips stop at the end of the song; bore and room ring out
        end = repeats * cycle
        evs.append(Event(end, 1.0, kind="rest"))
        track = tongue_track(song, tempo, repeats)
        c = player.controls(evs, settings, tail=0.0, tongue=track)
        return syn.render(c, reverb=reverb)

    # loop: one extra pass before and after the loop region
    total = repeats + 2
    evs = events(song, tempo, total)
    track = tongue_track(song, tempo, total)
    c = player.controls(evs, settings, tail=0.5, tongue=track)
    y = syn.render(c, reverb=reverb, normalize=False)
    n = int(round(cycle * fs))
    s, e = n, n * (1 + repeats)
    xf = int(crossfade * fs)
    out = y[s:e].copy()
    # crossfade the continuation after the end into the beginning
    w = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, xf))
    out[:xf] = out[:xf] * w + y[e:e + xf] * (1 - w)
    # short fade-out so the file does not end with a click when played once
    nf = int(fadeout * fs)
    if nf > 0:
        out[-nf:] *= 0.5 + 0.5 * np.cos(np.linspace(0, np.pi, nf))
    return out / (np.max(np.abs(out)) + 1e-12) * 0.89


def describe(song, tempo=None):
    """Human-readable listing of the timed syllables (for checking a song)."""
    lines = []
    for e in events(song, tempo):
        lines.append(f"{e.start:7.3f}s {e.dur * 1000:6.0f}ms  {e.label}")
    return "\n".join(lines)
