"""
Didgeridoo playing techniques and a small score language.

A *score* is a sequence of tokens; each token is one rhythmic event:

    [mods]syllable[:beats][@note]

``syllable``  optional consonant + optional vowel, e.g. ``doo``, ``di``, ``ga``,
              ``t``, ``rr``, ``wa``, ``ya``, ``ee``.
              consonants: t d k g p   (stops; p = lip/cheek pulse)
                          r           (single tongue flap, as in "ri-ti")
                          rr          (rolled tongue trill over the whole event)
                          y w         (glides from "ee" / "oo")
                          h ch s      (aspiration / fricatives)
              vowels:     a e i o u oo ee ae uh   (uh / none = neutral)
``mods``      ``!`` accent, ``*`` toot (``**`` second toot ...),
              ``~`` vocalise (sing into the instrument, pitch from ``@note``)
``beats``     duration in beats (default 1)
special       ``^``  circular breathing (sniff + cheek push), ``.`` rest,
              ``-``  hold the previous event for one more beat

Header lines ``name = value`` set ``tempo`` (BPM), ``voice`` (default note
for ``~``), ``swing`` and ``humanize``.

Example::

    tempo = 100
    doo:2 di:.5 ga:.5 di:.5 ga:.5 ^ *too:1 rr:1 ~oo:2@A3 wa ya !t:.5 k:.5
"""

import re
from dataclasses import dataclass, field

import numpy as np

from .model import ModelParams
from .synth import Controls
from .vocaltract import VOWELS, CONSONANTS, VocalTract

VOWEL_ALIASES = {"a": "a", "aa": "a", "e": "e", "i": "i", "ee": "i", "o": "o", "oh": "o",
                 "u": "u", "oo": "u", "ae": "ae", "wa": "ae", "uh": "neutral", "": "neutral",
                 "ai": "e", "ow": "o", "ou": "u"}
CONS_ORDER = ["ch", "rr", "r", "t", "d", "k", "g", "p", "y", "w", "h", "s"]

NOTE_RE = re.compile(r"^([A-Ga-g])([#b]?)(-?\d)$")


def note_to_freq(note):
    m = NOTE_RE.match(note)
    if not m:
        raise ValueError(f"bad note {note!r}")
    names = {"c": 0, "d": 2, "e": 4, "f": 5, "g": 7, "a": 9, "b": 11}
    n = names[m.group(1).lower()] + {"#": 1, "b": -1, "": 0}[m.group(2)]
    octave = int(m.group(3))
    midi = 12 * (octave + 1) + n
    return 440.0 * 2 ** ((midi - 69) / 12)


@dataclass
class Event:
    start: float              # seconds
    dur: float                # seconds
    consonant: str = ""
    vowel: str = "neutral"
    accent: bool = False
    toot: int = 0
    voice: float = 0.0        # Hz, 0 = no vocalisation
    kind: str = "syllable"    # syllable | breath | rest
    coda: str = ""            # syllable end: "ng" (dong) or "n"

    @property
    def label(self):
        if self.kind != "syllable":
            return self.kind
        parts = []
        if self.toot:
            parts.append(f"toot{self.toot}")
        if self.voice:
            parts.append("voice")
        if self.accent:
            parts.append("accent")
        if self.consonant:
            parts.append(self.consonant)
        parts.append(self.vowel)
        if self.coda:
            parts.append(self.coda)
        return "+".join(parts)


def parse_syllable(s):
    cons = ""
    for c in CONS_ORDER:
        if s.startswith(c):
            cons = c
            s = s[len(c):]
            break
    # tolerate doubled vowels written as "doo" -> d + oo, "dii" -> d + ii
    if s not in VOWEL_ALIASES:
        if len(s) > 1 and len(set(s)) == 1 and s[0] in VOWEL_ALIASES:
            s = s[0] if s[0] not in ("o", "e") else s[:2]
        else:
            raise ValueError(f"unknown vowel {s!r}")
    return cons, VOWEL_ALIASES[s]


TOKEN_RE = re.compile(r"^(?P<mods>[!*~]*)(?P<syl>[a-z]*|\^|\.|-)(?::(?P<dur>[0-9]*\.?[0-9]+))?"
                      r"(?:@(?P<note>[A-Ga-g][#b]?-?\d))?$")


def parse_score(text):
    """Parse a score; returns (events, settings)."""
    settings = {"tempo": 100.0, "voice": "A3", "swing": 0.0, "humanize": 1.0}
    tokens = []
    for line in text.splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        if "=" in line and not line.startswith(("!", "*", "~")):
            k, v = [x.strip() for x in line.split("=", 1)]
            settings[k] = v if k == "voice" else float(v)
            continue
        tokens += line.split()

    beat = 60.0 / float(settings["tempo"])
    events, t = [], 0.0
    for tok in tokens:
        m = TOKEN_RE.match(tok)
        if not m:
            raise ValueError(f"cannot parse token {tok!r}")
        mods, syl = m.group("mods"), m.group("syl")
        beats = float(m.group("dur")) if m.group("dur") else (0.5 if syl == "^" else 1.0)
        dur = beats * beat
        if syl == "-":
            if events:
                events[-1].dur += dur
            t += dur
            continue
        if syl == ".":
            events.append(Event(t, dur, kind="rest"))
        elif syl == "^":
            events.append(Event(t, dur, kind="breath"))
        else:
            cons, vowel = parse_syllable(syl)
            voice = 0.0
            if "~" in mods:
                voice = note_to_freq(m.group("note") or settings["voice"])
            events.append(Event(t, dur, cons, vowel, "!" in mods, mods.count("*"), voice))
        t += dur
    return events, settings


# ---------------------------------------------------------------------------
# controls
def _smooth(x, rate, tau):
    """Critically damped (two one-pole passes, forward) smoothing with time constant tau."""
    if tau <= 0:
        return x
    a = np.exp(-1.0 / (tau * rate))
    y = np.array(x, dtype=float, copy=True)
    for _ in range(2):
        acc = y[0]
        for i in range(len(y)):
            acc = a * acc + (1 - a) * y[i]
            y[i] = acc
    return y


def _smooth_fast(x, rate, tau):
    """Vectorised version of _smooth using scipy.signal.lfilter."""
    from scipy.signal import lfilter
    if tau <= 0:
        return x
    a = np.exp(-1.0 / (tau * rate))
    zi = np.atleast_1d(x[0] * a)
    y, _ = lfilter([1 - a], [1, -a], x, zi=zi * 1.0)
    y, _ = lfilter([1 - a], [1, -a], y, zi=np.atleast_1d(y[0] * a))
    return y


def _noise(rng, n, rate, tau):
    """Smoothed noise with unit standard deviation (steady state from the start)."""
    from scipy.signal import lfilter
    warm = int(5 * tau * rate)
    a = np.exp(-1.0 / (tau * rate))
    x = rng.standard_normal(n + warm)
    for _ in range(2):
        x = lfilter([1 - a], [1, -a], x)
    x = x[warm:]
    return x / (np.std(x) + 1e-12)


def _bump(t, t0, t_rise, t_hold, t_fall):
    """Raised-cosine envelope: rise to 1 at t0, hold, fall."""
    out = np.zeros_like(t)
    r = (t >= t0 - t_rise) & (t < t0)
    out[r] = 0.5 - 0.5 * np.cos(np.pi * (t[r] - (t0 - t_rise)) / t_rise)
    h = (t >= t0) & (t < t0 + t_hold)
    out[h] = 1.0
    f = (t >= t0 + t_hold) & (t < t0 + t_hold + t_fall)
    out[f] = 0.5 + 0.5 * np.cos(np.pi * (t[f] - t0 - t_hold) / t_fall)
    return out


@dataclass
class Player:
    """Turns events into control signals for a given bore.

    The player's habits (timing of the gestures, how deep a stop consonant
    interrupts the drone, ...) are parameters of this class.
    """
    bore: object
    params: ModelParams = field(default_factory=ModelParams)
    rate: float = 500.0
    base_vowel: str = "neutral"
    stop_dur: float = 0.025        # tongue closure of unvoiced stops, s
    stop_depth: float = 0.3        # extra lip-amplitude reduction during a t/k closure
    voiced_depth: float = 0.1      # ... during a d/g closure
    back_block: float = 0.85       # extra flow blockage of k/g closures (see _consonant)
    trill_rate: float = 26.0       # Hz, rolled r
    vibrato_cents: float = 4.0     # slow random pitch wander
    glide: float = 0.05            # time constant of mouth-shape transitions, s
    seed: int = 0

    def controls(self, events, settings=None, tail=0.4, tongue=None):
        """Control signals for a list of events (times in seconds).

        ``tongue`` optionally overrides the mouth shape: a callable that maps
        an array of times to an (n, 4) array of vocal-tract targets, NaN rows
        where the syllables decide.
        """
        settings = settings or {}
        hum = float(settings.get("humanize", 1.0))
        rng = np.random.default_rng(self.seed)
        end = max((e.start + e.dur for e in events), default=1.0) + tail
        p = self.params
        f_drone = self.bore.drone * 2 ** (p.pitch_cents / 1200)
        c = Controls.constant(end, self.rate, f_drone, p, VOWELS[self.base_vowel])
        n, rate = c.n, c.rate
        t = np.arange(n) / rate

        # ---- targets (piecewise constant), then smoothed
        vt_t = np.tile(VOWELS[self.base_vowel].as_array(), (n, 1))
        f0_t = np.full(n, f_drone)
        amp_t = np.ones(n)
        oq_t = np.full(n, p.oq)
        voice_t = np.zeros(n)
        vf_t = np.full(n, note_to_freq(settings.get("voice", "A3")) if isinstance(settings.get("voice", "A3"), str) else 220.0)

        jit = lambda s: s * hum
        for e in events:
            i0 = int(max(0, e.start + jit(rng.normal(0, 0.004))) * rate)
            i1 = int(min(end, e.start + e.dur) * rate)
            sl = slice(i0, i1)
            if e.kind == "rest":
                amp_t[sl] = 0.0
                c.labels.append((e.start, e.start + e.dur, "rest"))
                continue
            if e.kind == "breath":
                continue  # gesture applied below
            vt_t[sl] = VOWELS[e.vowel].as_array()
            if e.toot:
                f0_t[sl] = self.bore.toot(e.toot) * 2 ** ((p.pitch_cents + p.toot_cents) / 1200)
                amp_t[sl] *= 1.1
                oq_t[sl] -= 0.08
            if e.voice:
                voice_t[sl] = 1.0
                vf_t[sl] = e.voice
                vt_t[sl, 3] = 0.15         # adducted glottis while phonating
                c.gestures.append((e.start, e.start + e.dur, "voice"))
            if e.toot:
                c.gestures.append((e.start, e.start + e.dur, f"toot{e.toot}"))
            c.labels.append((e.start, e.start + e.dur, e.label))

        if tongue is not None:
            over = np.asarray(tongue(t), dtype=float)
            ok = np.all(np.isfinite(over), axis=1)
            vt_t[ok] = over[ok]
        c.vt = np.stack([_smooth_fast(vt_t[:, k], rate, self.glide) for k in range(4)], 1)
        c.f0 = _smooth_fast(f0_t, rate, 0.012)
        c.amp = _smooth_fast(amp_t, rate, 0.010)
        c.oq = _smooth_fast(oq_t, rate, 0.015)
        c.voice = _smooth_fast(voice_t, rate, 0.025)
        c.voice_f0 = _smooth_fast(vf_t, rate, 0.03)

        # ---- gestures (consonants, accents, breathing), applied on top
        for e in events:
            t0 = e.start
            if e.kind == "breath":
                self._circular_breath(c, t, t0, e.dur)
                continue
            if e.kind != "syllable":
                continue
            if e.accent:
                env = _bump(t, t0, 0.005, 0.02, 0.15)
                c.amp *= 1 + 0.6 * env
                c.oq -= 0.12 * env
                c.gestures.append((t0, t0 + 0.1, "accent"))
            if e.consonant:
                self._consonant(c, t, t0, e)
            if e.coda:
                self._coda(c, t, e)

        # ---- humanisation: slow pitch wander + amplitude shimmer
        c.f0 = c.f0 * 2 ** (self.vibrato_cents * hum * _noise(rng, n, rate, 0.25) / 1200)
        c.amp *= 1 + 0.04 * hum * _noise(rng, n, rate, 0.05)
        # the mouth is never perfectly still: slow wander of tongue and jaw
        for k, size in [(0, 0.02), (1, 0.02), (2, 0.03)]:
            c.vt[:, k] += size * hum * _noise(rng, n, rate, 0.3)
        # fade in/out
        c.amp *= np.clip(t / 0.08, 0, 1) * np.clip((end - t) / 0.15, 0, 1)
        c.oq = np.clip(c.oq, 0.2, 0.95)
        c.vt = np.clip(c.vt, 0, 1)
        return c

    # ------------------------------------------------------------------ gestures
    def _consonant(self, c, t, t0, e):
        info = CONSONANTS[e.consonant]
        kind, place = info["kind"], info["tongue_pos"]
        if kind == "stop" and e.consonant == "p":
            # lip/cheek pulse: short amplitude spike and brighter lips
            env = _bump(t, t0, 0.004, 0.012, 0.03)
            c.amp *= 1 + 1.0 * env
            c.oq -= 0.15 * env
            c.gestures.append((t0 - 0.004, t0 + 0.04, "pulse"))
            return
        if kind == "stop":
            dur = self.stop_dur * (0.7 if info["voiced"] else 1.0)
            depth = self.voiced_depth if info["voiced"] else self.stop_depth
            if place >= 0.3:
                # a closure at the velum leaves a large front cavity, which the
                # vocal-tract impedance treats as a resonator instead of a
                # blockage; the stopped airflow must come from the lips instead
                depth += (1 - depth) * self.back_block
            # closure ends at t0 (release on the beat)
            close = _bump(t, t0 - dur, 0.012, dur, 0.012)
            move = _bump(t, t0 - dur, 0.03, dur, 0.04)
            c.vt[:, 0] += (place - c.vt[:, 0]) * move
            c.vt[:, 1] += (1.0 - c.vt[:, 1]) * close
            c.amp *= 1 - depth * close
            burst = _bump(t, t0 + 0.004, 0.003, 0.004, 0.025 if not info["voiced"] else 0.012)
            c.frication += burst * (0.35 if not info["voiced"] else 0.12)
            c.fric_freq = np.where(burst > 0, 4500 if place < 0.3 else 2200, c.fric_freq)
            # release: brief brightening
            rel = _bump(t, t0 + 0.005, 0.005, 0.0, 0.06)
            c.oq -= 0.06 * rel
            label = ("stop_front" if place < 0.3 else "stop_back") + ("_voiced" if info["voiced"] else "")
            c.gestures.append((t0 - dur - 0.012, t0 + 0.012, label))
        elif kind == "tap":
            # one quick flap of the tongue tip against the alveolar ridge
            close = _bump(t, t0 - 0.008, 0.006, 0.006, 0.008)
            c.vt[:, 0] += (place - c.vt[:, 0]) * close
            c.vt[:, 1] += (0.97 - c.vt[:, 1]) * close
            c.amp *= 1 - 0.15 * close
            c.gestures.append((t0 - 0.014, t0 + 0.008, "tap"))
        elif kind == "trill":
            span = e.dur if e.consonant == "rr" else min(e.dur, 0.12)
            gate = _bump(t, t0 + 0.01, 0.01, max(span - 0.03, 0.0), 0.02)
            mod = 0.5 + 0.5 * np.sin(2 * np.pi * self.trill_rate * (t - t0))
            c.vt[:, 0] += (place - c.vt[:, 0]) * gate
            c.vt[:, 1] += (0.97 - c.vt[:, 1]) * gate * mod
            c.amp *= 1 - 0.55 * gate * mod
            c.gestures.append((t0, t0 + span, "trill"))
        elif kind == "glide":
            start = VOWELS["i"] if e.consonant == "y" else VOWELS["u"]
            env = _bump(t, t0, 0.01, 0.02, 0.09)
            c.vt += (start.as_array()[None, :] - c.vt) * env[:, None]
            c.gestures.append((t0 - 0.01, t0 + 0.11, "glide"))
        elif kind == "aspirate":
            env = _bump(t, t0, 0.01, 0.03, 0.05)
            c.frication += 0.18 * env
            c.fric_freq = np.where(env > 0, 1800, c.fric_freq)
            c.amp *= 1 - 0.3 * env
            c.gestures.append((t0 - 0.01, t0 + 0.08, "aspirate"))
        elif kind == "fricative":
            env = _bump(t, t0, 0.015, 0.06, 0.03)
            c.vt[:, 0] += (place - c.vt[:, 0]) * env
            c.vt[:, 1] += (0.93 - c.vt[:, 1]) * env
            c.frication += 0.3 * env
            c.fric_freq = np.where(env > 0, 5000 if e.consonant == "s" else 3200, c.fric_freq)
            c.amp *= 1 - 0.4 * env
            c.gestures.append((t0 - 0.015, t0 + 0.09, "fricative"))

    def _coda(self, c, t, e):
        """Syllable end: the tongue moves towards a closure (velum for "ng",
        alveolar ridge for "n") during the last third of the syllable, without
        a release burst; the drone keeps sounding (nasal ending)."""
        info = CONSONANTS[e.coda]
        d = min(e.dur * 0.35, 0.12)
        t_end = e.start + e.dur
        env = _bump(t, t_end - 0.01, d, 0.0, 0.03)
        c.vt[:, 0] += (info["tongue_pos"] - c.vt[:, 0]) * env
        c.vt[:, 1] += (0.9 - c.vt[:, 1]) * env
        c.gestures.append((t_end - d, t_end, f"coda_{e.coda}"))

    def _circular_breath(self, c, t, t0, dur):
        """Circular breathing: the back of the tongue seals the mouth, the
        cheeks push the stored air through the lips while the player sniffs
        in through the nose. Audible as a short timbre change (front cavity
        only), an accent from the cheek push and the nasal inhalation noise."""
        d = max(dur, 0.2)
        env = _bump(t, t0 + 0.03, 0.04, d - 0.12, 0.06)
        seal = VocalTract(0.85, 0.97, 0.15, 0.5).as_array()
        c.vt += (seal[None, :] - c.vt) * env[:, None]
        push = _bump(t, t0 + 0.06, 0.03, 0.02, 0.12)
        c.amp *= (1 + 0.35 * push) * (1 - 0.15 * env)
        c.oq -= 0.05 * push
        c.f0 *= 2 ** (-25 * env / 1200)
        c.sniff += 0.05 * _bump(t, t0 + 0.05, 0.05, d - 0.15, 0.06)
        c.labels.append((t0, t0 + dur, "circular_breath"))
        c.gestures.append((t0 + 0.03 - 0.04, t0 + 0.03 + d - 0.12 + 0.06, "circular_breath"))


def score_controls(bore, score, params=ModelParams(), **player_kw):
    events, settings = parse_score(score)
    player = Player(bore, params, **player_kw)
    return player.controls(events, settings)


# ---------------------------------------------------------------------------
# random performances (training data for the technique detector)
CELLS = {
    "drone": ["doo:2", "oo:1", "uh:2", "ee:1 oo:1", "wa:1 ya:1", "oo:.5 ee:.5", "o:1", "a:1", "e:1",
              "ee:2", "ae:1"],
    "stops": ["di:.5 ga:.5", "ti:.5 ka:.5", "da:.5 ga:.5", "t:.5 k:.5", "di:.25 di:.25 ga:.5",
              "ka:.5 ti:.5", "ti:.25 ti:.25 ka:.5", "do:.5 go:.5", "ta:1", "ga:1", "du:.5 du:.5"],
    "trill": ["rr:1", "rr:.5", "d:.25 rr:.75", "rr:2"],
    "toot": ["*too:1", "*ta:.5", "*too:.5 doo:.5"],
    "voice": ["~oo:2@{note}", "~a:1@{note}", "~ee:1@{note}", "~oo:1@{note} ~oo:1@{note2}"],
    "accent": ["!t:.5", "!da:.5", "!ka:.5", "!oo:1"],
    "fric": ["ch:.5", "s:.5", "h:.5", "chu:1"],
    "pulse": ["p:.5", "pa:.5 pa:.5"],
    "rest": [".:.5 doo:1", ".:1", "doo:.5 .:.5", "!oo:.75 .:.25", ".:.25 ee:.75"],
}
CELL_WEIGHTS = {"drone": 3, "stops": 4, "trill": 1, "toot": 1, "voice": 1, "accent": 1, "fric": 1,
                "pulse": 0.5, "rest": 0.7}
VOICE_NOTES = ["E2", "G2", "A2", "C3", "D3", "E3", "G3", "A3", "B3", "C4", "D4", "E4"]


def random_score(rng, duration=20.0, tempo=None, breath_every=(2.0, 5.0), n_toots=1):
    """A random but idiomatic sequence of techniques (used as training data)."""
    tempo = tempo or float(rng.uniform(80, 140))
    beat = 60 / tempo
    names = list(CELLS)
    w = np.array([CELL_WEIGHTS[k] for k in names], float)
    w /= w.sum()
    out, t = [], 0.0
    next_breath = rng.uniform(*breath_every)
    while t < duration:
        if t >= next_breath:
            out.append("^:0.5")
            t += 0.5 * beat
            next_breath = t + rng.uniform(*breath_every)
            continue
        kind = names[rng.choice(len(names), p=w)]
        cell = CELLS[kind][rng.integers(len(CELLS[kind]))]
        if kind == "toot" and n_toots > 1 and rng.random() < 0.3:
            cell = cell.replace("*", "**", 1)
        cell = cell.format(note=VOICE_NOTES[rng.integers(len(VOICE_NOTES))],
                           note2=VOICE_NOTES[rng.integers(len(VOICE_NOTES))])
        out.append(cell)
        t += sum(float(tok.split(":")[1].split("@")[0]) if ":" in tok else 1.0
                 for tok in cell.split()) * beat
    return f"tempo = {tempo:.1f}\n" + " ".join(out)


def random_player(rng, bore, params=None):
    """Randomised player habits and recording conditions."""
    p = params or ModelParams()
    p = p.with_(oq=float(rng.uniform(0.35, 0.6)), skew=float(rng.uniform(0.7, 0.9)),
                zs_mpa=float(np.exp(rng.uniform(np.log(0.5), np.log(2.0)))),
                noise_db=float(rng.uniform(-46, -30)), pitch_cents=float(rng.uniform(-30, 20)))
    player = Player(bore, p, base_vowel=str(rng.choice(list(VOWELS))),
                    stop_dur=float(rng.uniform(0.015, 0.045)), stop_depth=float(rng.uniform(0.0, 0.6)),
                    voiced_depth=float(rng.uniform(0.0, 0.3)), trill_rate=float(rng.uniform(20, 32)),
                    vibrato_cents=float(rng.uniform(2, 8)), seed=int(rng.integers(1 << 30)))
    return player
