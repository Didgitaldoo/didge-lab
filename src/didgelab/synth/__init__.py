"""
Physically informed didgeridoo synthesizer: bore geometry -> sound.

Input is a bore geometry (list of [x_mm, diameter_mm], the DidgeLab format).
DidgeLab computes the input impedance; the synthesizer adds the transfer
function of the bore, a beating-lip source and the player's vocal tract, and
renders playing techniques written in a small score language or in ``.didge``
song files.

Quick start::

    from didgelab.synth import render_song_loop
    y, fs = render_song_loop([[0, 32], [1500, 75]], "tiki-du")
"""

from pathlib import Path

from ..sim.sim_interface import compute_moist_air_properties
from .bore import Bore
from .model import ModelParams, filter_gain, harmonic_spectrum
from .vocaltract import VocalTract, VOWELS
from .synth import Synthesizer, Controls
from .techniques import Player, parse_score, score_controls, random_score
from . import song as _song

__all__ = ["Bore", "ModelParams", "VocalTract", "VOWELS", "Synthesizer", "Controls", "Player",
           "parse_score", "score_controls", "random_score", "synthesize", "filter_gain",
           "harmonic_spectrum", "SONGS", "render_song_loop"]

DEFAULT_SCORE = """tempo = 100
doo:2 di:.5 ga:.5 di:.5 ga:.5 ^ doo:1 ti:.5 ka:.5 ti:.5 ka:.5 *too:1 doo:1 rr:1 doo:1 wa:1 ya:1 ee:1 oo:1 ^ ~oo:2@C4 doo:2
"""

SONGS_DIR = Path(__file__).parent / "songs"

# Songs that can be rendered by name (fixed whitelist: no paths from callers).
SONGS = {
    "dong-tiki-ritititi": SONGS_DIR / "dong-tiki-ritititi.didge",
    "tiki-du": SONGS_DIR / "tiki-du.didge",
    "drone-back-to-front": SONGS_DIR / "drone-back-to-front.didge",
}


def synthesize(geometry, score=DEFAULT_SCORE, fs=48000, params=None, reverb=0.25, seed=0,
               **player_kw):
    """Render a score on a bore. Returns (audio, fs)."""
    params = params or ModelParams()
    bore = Bore(geometry, loss_factor=params.loss_factor)
    c = score_controls(bore, score, params, seed=seed, **player_kw)
    y = Synthesizer(bore, params, fs=fs, seed=seed).render(c, reverb=reverb)
    return y, fs


def render_song_loop(geometry, song, temperature=28.0, tempo=100.0, fs=48000):
    """Render one seamless loop (one pass of the song) on a bore.

    Parameters
    ----------
    geometry : list of [x_mm, diameter_mm] or ``didgelab.Geo``
    song : name of a song in ``SONGS``
    temperature : air temperature in degrees Celsius (breath humidity 100 %)
    tempo : quarter notes per minute, overrides the tempo of the song file
    fs : sample rate in Hz (must be a multiple of the synth control rate, e.g. 48000)

    Returns ``(audio, fs)`` where audio is a mono float array whose length is
    exactly one pass of the song and which loops without a click.
    """
    if song not in SONGS:
        raise ValueError(f"unknown song {song!r}, available: {sorted(SONGS)}")
    constants = compute_moist_air_properties(temp_celsius=temperature)
    bore = Bore(geometry, constants=constants)
    s = _song.load(SONGS[song])
    return _song.render(s, bore, tempo=tempo, repeats=1, loop=True, fs=fs,
                         fadeout=0.0), fs
