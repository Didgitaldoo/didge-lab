"""
Access to the Didge Archive (https://github.com/Didgitaldoo/didge-archive).

The archive is located via (in this order) the ``path`` argument, the
``DIDGE_ARCHIVE`` environment variable, ``./didge-archive``, or a clone in
``~/.cache/didgelab/didge-archive`` (created with ``git clone`` on demand).

Ported from didge-synth (``didgesynth.archive``).
"""

import json
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

URL = "https://github.com/Didgitaldoo/didge-archive.git"

# Corrections of the archive index. Tamaki2 points to the Open Didgeridoo
# geometry, but the archive also contains tamaki2.json. The simulated drone of
# tamaki2.json matches the recording (-10 cents) while opendidgeridoo.json is
# 45 cents off, so we use tamaki2.json.
GEOMETRY_OVERRIDES = {"Tamaki2": "geometries/tamaki2.json"}


def locate(path=None, download=True):
    candidates = [path, os.environ.get("DIDGE_ARCHIVE"), "didge-archive",
                  Path.home() / ".cache" / "didgelab" / "didge-archive"]
    for c in candidates:
        if c and (Path(c) / "didge-archive.json").exists():
            return Path(c)
    if not download:
        raise FileNotFoundError("Didge Archive not found; set DIDGE_ARCHIVE")
    target = Path.home() / ".cache" / "didgelab" / "didge-archive"
    target.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "clone", "--depth", "1", URL, str(target)], check=True)
    return target


@dataclass
class Instrument:
    name: str
    geometry: list
    samples: dict = field(default_factory=dict)   # kind -> absolute path
    shape: str = ""
    geometry_file: str = ""


def load(path=None):
    root = locate(path)
    index = json.loads((root / "didge-archive.json").read_text())
    out = []
    for d in index:
        gfile = GEOMETRY_OVERRIDES.get(d["name"], d["geometry"])
        geo = json.loads((root / gfile).read_text())
        samples = {k: str(root / v) for k, v in d.get("audio-samples", {}).items()
                   if (root / v).exists()}
        out.append(Instrument(d["name"], geo, samples, d.get("shape", ""), gfile))
    return out


def get(name, path=None):
    for inst in load(path):
        if inst.name.lower() == name.lower():
            return inst
    raise KeyError(name)
