# Finding playing techniques in didgeridoo recordings

`didgelab.synth.transcribe` takes a recording and the bore of the instrument and
reports what the player did: toots, stop consonants (t/d at the front of the
mouth, k/g at the back), rolled-r trills, vocalisation, circular breathing,
fricatives, and where the tongue is. It uses the physical model of the
synthesizer to interpret the sound ("analysis by synthesis").

The method and its evaluation are described in detail in
[synthesizer.md](synthesizer.md), §8 and §9. A runnable walk-through with a
synthetic and a real recording is the notebook
[examples/technique_detection.ipynb](examples/technique_detection.ipynb).

## 1. Usage

```python
from didgelab.synth import Bore, archive, load_audio, transcribe
from didgelab.synth.transcribe import plot_transcription

inst = archive.get("Tamaki1")                       # or any [[x_mm, d_mm], ...] geometry
x, fs = load_audio(inst.samples["song"], 48000)
fa, segments = transcribe(x, fs, Bore(inst.geometry))
for s in segments:
    print(f"{s.start:6.2f}-{s.end:6.2f} s  {s.label}  {s.info}")
plot_transcription(x, fs, fa, segments, "transcription.png", title="Tamaki1")
```

From the command line:

```bash
didgelab-synth analyze recording.wav --geometry Tamaki1 --plot transcription.png --json events.json
```

`--geometry` is a geometry JSON file or the name of a Didge Archive instrument.
The bore must be known: the analysis relies on its resonances. If the bore is
unknown, its impedance can be estimated from the recording first
([impedance_from_sound.md](impedance_from_sound.md)).

The trained detector needs `scikit-learn` and `joblib`
(`pip install didgelab[synth]`). Without them, or if the model file cannot be
loaded, `transcribe` uses a simple rule-based detector, which is much less
reliable.

## 2. How it works

Every 10 ms frame of the recording is analysed (`detect.analyse`):

1. **Which resonance do the lips play?** The lips of a player lock to a
   resonance of the bore, so the lip frequency is searched only within ±90 cents
   of the DidgeLab resonances. The lowest resonance whose harmonics are within
   4 dB of the best one wins. This avoids octave errors. Playing on the 2nd or a
   higher resonance is a toot.
2. **Where is the tongue?** The dominant formant of the frame (the strongest
   peak of the spectral envelope between 500 Hz and 4 kHz, relative to the
   long-term envelope) is translated into a **tongue index** from 0 (front) to
   1 (back) with the vocal-tract model of the same bore.
3. **Features** per frame: level and its changes, lip mode, energy between the
   harmonics, high-band noise, 15–45 Hz amplitude modulation, band energies,
   formant and tongue index, with ±100 ms context.

The techniques are then detected by a classifier (`learn.Detector`, one
gradient-boosted tree ensemble per technique) that has **never seen a real
recording**. It is trained on synthetic performances: random scores played by
random players on random bores, under random recording conditions. Because the
synthesizer knows what it played, every frame has exact labels.

Two physical plausibility checks are applied afterwards: a vocalisation must show
sum and difference tones between the harmonics, and a trill must show a periodic
15–45 Hz modulation of the level.

## 3. Accuracy

Tested on 32 synthetic performances on the eight Didge Archive bores, which the
detector never saw during training
([results/synth/detector_eval.json](results/synth/detector_eval.json)):

| technique | precision | recall | F1 |
|---|---|---|---|
| stop consonants (±60 ms) | 0.98 | 0.97 | 0.97 |
| place of articulation (t/d vs. k/g) | | | accuracy 0.955 |
| rolled-r trill | 0.95 | 0.98 | 0.97 |
| vocalisation | 0.94 | 1.00 | 0.97 |
| circular breathing | 1.00 | 0.99 | 0.99 |
| fricatives / aspiration | 0.84 | 0.77 | 0.80 |
| toots (lip tracker) | 0.99 | 1.00 | 0.99 |

These numbers say how well the techniques **as modelled** are recognised on
unseen instruments. On real recordings, where players execute the techniques
differently, they are an upper bound.

On the real recordings of the Didge Archive (§9 of the article,
[results/synth/recordings_summary.md](results/synth/recordings_summary.md)):

* the drone and the **toots** are found reliably; toots are played 27–40 cents
  below the 2nd resonance;
* **tongue movements** between front and back (formant ≈1–3 kHz) are the dominant
  technique in every recording;
* **vocalisation** is clearly identified in one recording;
* **stop consonants and circular breathing** are uncertain: real tongue stops are
  subtler than in the model. Read these detections as hypotheses.

## 4. Retraining the detector

The trained model is stored in `src/didgelab/synth/data/detector.joblib`. Like
every scikit-learn model it is tied to the scikit-learn version it was saved
with. If it does not load, `transcribe` warns and falls back to the rule-based
detector. Retrain it with:

```bash
export DIDGE_ARCHIVE=/path/to/didge-archive        # test bores (cloned on demand otherwise)
python scripts/synth/train_detector.py             # from the didge-lab root, ~1.5 h on 3 cores
```

This renders 80 synthetic training performances and 32 test performances (20 s
each), caches them in `outputs/synthetic.pkl`, trains the model and writes
`doc/results/synth/detector_eval.json`. `--eval-only` evaluates the saved model.
`scripts/synth/analyze_recordings.py` transcribes all songs of the archive and
writes the figures `doc/figures/transcription-*.png` and the tables in
`doc/results/synth/`.
