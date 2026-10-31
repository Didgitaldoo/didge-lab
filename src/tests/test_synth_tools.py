"""Archive access, transcription and the command line interface of didgelab.synth."""
import json
import os
from pathlib import Path

import numpy as np
import pytest

from didgelab.synth import Bore, ModelParams, Synthesizer, archive, load_audio
from didgelab.synth.cli import main
from didgelab.synth.techniques import score_controls
from didgelab.synth.transcribe import transcribe

GEO = [[0, 32], [800, 32], [1100, 40], [1450, 75]]
ARCHIVE = os.environ.get("DIDGE_ARCHIVE")


def test_load_audio_reads_wav(tmp_path):
    from scipy.io import wavfile
    fs = 16000
    y = 0.5 * np.sin(2 * np.pi * 100 * np.arange(fs) / fs)
    wavfile.write(tmp_path / "a.wav", fs, (y * 32767).astype(np.int16))
    x, sr = load_audio(tmp_path / "a.wav", fs=48000)
    assert sr == 48000 and abs(len(x) - 48000) <= 3 and abs(np.abs(x).max() - 0.5) < 0.01


@pytest.mark.skipif(not ARCHIVE, reason="set DIDGE_ARCHIVE to a clone of the Didge Archive")
def test_archive_instruments_and_override():
    inst = archive.load(ARCHIVE)
    names = [i.name for i in inst]
    assert "Arusha" in names and "Tamaki2" in names
    assert archive.get("Tamaki2", ARCHIVE).geometry == json.loads(
        (Path(ARCHIVE) / "geometries" / "tamaki2.json").read_text())


def test_transcribe_finds_toot():
    b = Bore(GEO)
    p = ModelParams()
    c = score_controls(b, "tempo = 60\ndoo:2 *too:1 doo:2", p)
    y = Synthesizer(b, p, seed=0).render(c, reverb=0.0)
    fa, segs = transcribe(y, 48000, b)
    toots = [s for s in segs if s.label.startswith("toot")]
    assert toots and all(1.5 < 0.5 * (s.start + s.end) < 3.5 for s in toots)


def test_cli_info_and_synth(tmp_path, capsys):
    geo = tmp_path / "geo.json"
    geo.write_text(json.dumps(GEO))
    main(["info", str(geo)])
    assert "freq/Hz" in capsys.readouterr().out
    out = tmp_path / "out.wav"
    main(["synth", str(geo), "-s", "tempo = 120\ndoo:1", "-o", str(out)])
    x, fs = load_audio(out)
    assert fs == 48000 and len(x) > fs // 2
