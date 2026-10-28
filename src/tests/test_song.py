import numpy as np
import pytest

from didgelab.synth import song as S
from didgelab.synth import Bore

GEO = [[0, 32], [1500, 72]]


def test_bar_division_and_syllables():
    sg = S.parse("tempo: 120\n[rhythm]\n| dong-tiki dong-tiki dong:1/8 tiki:1/8 ritititi |")
    ev = S.events(sg)
    labels = [e.label for e in ev]
    assert labels[:3] == ["d+o+ng", "t+i", "k+i"]
    assert labels[-4:] == ["r+i", "t+i", "t+i", "t+i"]
    beat = 0.5
    assert ev[0].dur == pytest.approx(beat / 2)        # dong: an eighth
    assert ev[1].dur == pytest.approx(beat / 4)        # ti: a sixteenth
    assert ev[6].start == pytest.approx(2 * beat)      # third beat: dong
    assert ev[-1].start + ev[-1].dur == pytest.approx(4 * beat)


def test_ties_rests_and_errors():
    sg = S.parse("[rhythm]\n| uh _ _ . |")
    ev = S.events(sg, tempo=60)
    assert ev[0].dur == pytest.approx(3.0) and ev[1].kind == "rest"
    with pytest.raises(S.SongError):
        S.parse("[rhythm]\n| doo:1/2 doo:1/2 doo |")       # no time left
    with pytest.raises(S.SongError):
        S.parse("[rhythm]\n| dxq |")
    with pytest.raises(S.SongError):
        S.parse("[rhythm]\n| doo |\n[tongue]\n| sideways |")


def test_tongue_lane_glide_and_hold():
    sg = S.parse("tempo: 60\n[rhythm]\n| uh _ _ _ |\n[tongue]\n| oo ~ ~ ee |")
    tr = S.tongue_track(sg)
    tp = tr(np.array([0.5, 1.0, 2.0, 3.5]))[:, 0]
    oo, ee = S.tongue_value("oo")[0], S.tongue_value("ee")[0]
    assert tp[0] == pytest.approx(oo) and tp[1] == pytest.approx(oo)   # held during beat 1
    assert ee < tp[2] < oo                                              # gliding
    assert tp[3] == pytest.approx(ee)
    sg = S.parse("tempo: 60\n[rhythm]\n| uh _ _ _ |\n[tongue]\n| oo . . ee |")
    tp = S.tongue_track(sg)(np.array([2.9]))[:, 0]
    assert tp[0] == pytest.approx(oo)                                   # hold until the jump
    sg = S.parse("[rhythm]\n| uh _ |\n[tongue]\n| auto ee |")
    assert np.all(np.isnan(S.tongue_track(sg, tempo=60)(np.array([0.5]))))


def test_render_one_shot_and_seamless_loop():
    sg = S.parse("tempo: 120\nhumanize: 0\n[rhythm]\n| dong-ti dong-ti |")
    b = Bore(GEO)
    y = S.render(sg, b)
    assert np.all(np.isfinite(y)) and len(y) > 48000 * 2
    yl = S.render(sg, b, repeats=2, loop=True, fadeout=0.0)
    assert len(yl) == round(2 * 2.0 * 48000)
    seam = np.abs(np.concatenate([yl[-2:], yl[:2]])).max()
    assert np.abs(np.diff(np.concatenate([yl[-1:], yl[:1]])))[0] < 0.05 and seam <= 1.0


def test_render_song_loop_length_and_whitelist():
    from didgelab.synth import render_song_loop, SONGS
    geometry = [[0, 32], [800, 32], [1500, 72]]
    y, fs = render_song_loop(geometry, "tiki-du", temperature=28.0, tempo=100.0, fs=48000)
    assert len(y) == round(4 * 60 / 100.0 * fs)       # one 4/4 bar at 100 bpm
    assert abs(y).max() <= 1.0
    assert "drone-back-to-front" in SONGS
    try:
        render_song_loop(geometry, "../etc/passwd")
    except ValueError:
        pass
    else:
        raise AssertionError("unknown song must be rejected")
