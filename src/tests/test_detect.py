import numpy as np

from didgelab.synth import Bore, ModelParams, Synthesizer
from didgelab.synth.detect import analyse, track_lip_f0
from didgelab.synth.techniques import score_controls

GEO = [[0, 30], [72.6, 24], [173, 26.7], [257.7, 36], [353, 24], [483, 30.3], [545, 36.2],
       [628, 36.5], [760.6, 25.7], [853, 26.6], [945.8, 37.3], [998, 29.3], [1119, 49.8],
       [1183.5, 55.5], [1296, 60.4], [1388.7, 77]]


def test_lip_tracker_finds_toot_and_drone():
    b = Bore(GEO)
    p = ModelParams()
    c = score_controls(b, "tempo = 60\ndoo:1 *too:1 doo:1", p)
    y = Synthesizer(b, p, seed=0).render(c, reverb=0.0)
    fa = analyse(y, 48000, b, p, invert=False)
    t = fa.t
    assert np.median(fa.mode[(t > 0.3) & (t < 0.9)]) == 0
    assert np.median(fa.mode[(t > 1.2) & (t < 1.8)]) == 1


def test_vt_inversion_recovers_tongue_position():
    b = Bore(GEO)
    p = ModelParams()
    c = score_controls(b, "tempo = 60\nee:2 oo:2", p)
    y = Synthesizer(b, p, seed=0).render(c, reverb=0.0)
    fa = analyse(y, 48000, b, p)
    t = fa.t
    front = np.nanmedian(fa.vt[(t > 0.5) & (t < 1.8), 0])
    back = np.nanmedian(fa.vt[(t > 2.5) & (t < 3.8), 0])
    assert front < back


def test_tongue_index_from_formant():
    # formants are measured relative to the long-term envelope, so the test
    # signal spends equal time at the front and at the back
    b = Bore(GEO)
    p = ModelParams()
    c = score_controls(b, "tempo = 60\nee:2 oo:2 ee:2 oo:2", p)
    y = Synthesizer(b, p, seed=0).render(c, reverb=0.0)
    fa = analyse(y, 48000, b, p, invert=False)
    t = fa.t
    front = (t > 0.6) & (t < 1.8) | (t > 4.6) & (t < 5.8)
    back = (t > 2.6) & (t < 3.8) | (t > 6.6) & (t < 7.8)
    assert np.nanmedian(fa.tongue_index[front]) < np.nanmedian(fa.tongue_index[back])
    assert np.nanmedian(fa.formant_hz[front]) > np.nanmedian(fa.formant_hz[back])



def test_rule_based_detector_uses_the_labels_of_the_learned_detector():
    from didgelab.synth.detect import detect
    from didgelab.synth.learn import CLASSES
    b = Bore([[0, 32], [800, 32], [1100, 40], [1450, 75]])
    p = ModelParams()
    c = score_controls(b, "tempo = 60\ndoo:2 *too:1 doo:2", p)
    y = Synthesizer(b, p, seed=0).render(c, reverb=0.0)
    segs = detect(analyse(y, 48000, b, p))
    stops = [s for s in segs if s.label.startswith("stop")]
    assert stops
    assert all(s.info["consonant"] in "tdkg" for s in stops)
    assert all(s.label in CLASSES or s.label.startswith("toot") for s in segs)
