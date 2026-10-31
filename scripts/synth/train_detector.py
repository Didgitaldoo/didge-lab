"""Train the technique detector on synthetic performances and evaluate it on
the Didge Archive geometries (instruments never seen during training).

    python scripts/synth/train_detector.py [--train 80] [--test-per-instrument 4]
"""
import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np

from didgelab.synth import archive
from didgelab.synth.learn import build_dataset, train, events_from_proba, evaluate_events, CLASSES

ap = argparse.ArgumentParser()
ap.add_argument("--train", type=int, default=80)
ap.add_argument("--test-per-instrument", type=int, default=4)
ap.add_argument("--duration", type=float, default=20.0)
ap.add_argument("--archive", default=None)
ap.add_argument("--cache", default="outputs/synthetic.pkl")
ap.add_argument("--out", default="src/didgelab/synth/data/detector.joblib")
ap.add_argument("--eval-only", action="store_true", help="evaluate the saved detector")
args = ap.parse_args()

cache = Path(args.cache)
if cache.exists():
    train_ex, test_ex, test_names = pickle.load(open(cache, "rb"))
else:
    t0 = time.time()
    train_jobs = [(1000 + i, None, args.duration) for i in range(args.train)]
    geos = {}
    for inst in archive.load(args.archive):
        geos.setdefault(inst.geometry_file, (inst.name, inst.geometry))
    test_jobs, test_names = [], []
    for k, (name, geo) in enumerate(geos.values()):
        for j in range(args.test_per_instrument):
            test_jobs.append((50000 + 100 * k + j, geo, args.duration))
            test_names.append(name)
    train_ex = build_dataset(train_jobs)
    print(f"train set {time.time() - t0:.0f}s", flush=True)
    test_ex = build_dataset(test_jobs)
    print(f"test set {time.time() - t0:.0f}s", flush=True)
    cache.parent.mkdir(parents=True, exist_ok=True)
    pickle.dump((train_ex, test_ex, test_names), open(cache, "wb"))

if args.eval_only:
    from didgelab.synth.learn import Detector
    det = Detector.load(args.out)
else:
    det = train(train_ex)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    det.save(args.out)

# frame-level and event-level evaluation on unseen instruments
from types import SimpleNamespace
frame = {}
segs_list = []
for X, Y, meta in test_ex:
    P = det.predict_proba(X)
    for k, c in enumerate(CLASSES):
        pred = P[:, k] > det.thresholds[k]
        st = frame.setdefault(c, [0, 0, 0])
        st[0] += int(np.sum(pred & (Y[:, k] == 1)))
        st[1] += int(np.sum(pred))
        st[2] += int(np.sum(Y[:, k] == 1))
    t = np.array(meta["t"])
    fa = SimpleNamespace(mode=np.array(meta["mode"]), f0=np.full(len(t), np.nan),
                         level_db=np.array(meta.get("level_db", np.zeros(len(t)))))
    segs_list.append(events_from_proba(P, t, det.thresholds, fa))
frame_f1 = {c: 2 * st[0] / max(st[1] + st[2], 1) for c, st in frame.items()}
ev = evaluate_events([e[2] for e in test_ex], segs_list)
res = {"frame_f1": frame_f1, "events": ev, "n_train": len(train_ex), "n_test": len(test_ex),
       "test_instruments": sorted(set(test_names))}
Path("doc/results/synth").mkdir(parents=True, exist_ok=True)
json.dump(res, open("doc/results/synth/detector_eval.json", "w"), indent=2)
print(json.dumps(res, indent=2))
