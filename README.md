# DidgeLab

## Table of contents

[Table of contents](#table-of-contents)
[1. Introduction](#1-introduction)
[2. Related works](#2-related-works)
[3. Getting Started](#3-getting-started)
[3.1 Installation](#31-installation)
[3.2 Documentation](#32-documentation)
[4. Developer Instructions](#4-developer-instructions)
[4.1 Compile from source](#41-compile-from-source)
[4.2 Unit Tests (pytest)](#42-unit-tests-pytest)
[4.3 Building the Python package](#43-building-the-python-package)
[4.4 Build Documentation](#44-build-documentation)
[5. Source code overview](#5-source-code-overview)
[6. Licensing](#6-licensing)

## 1. Introduction

DidgeLab is a free open-source toolkit to compute didgeridoo geometries. Traditionally, building a didgeridoo is a random process: builders know how geometry influences the sound, but the exact sonic properties can only be determined after the instrument is built. DidgeLab helps by first defining the desired sound and then computing a geometry that (in simulation) achieves it.

It provides:

1. **Acoustical simulation** — Compute resonant frequencies and impedance spectra for a given didgeridoo geometry. This is similar in spirit to Didgmo and DidjiImp.
2. **Computational evolution** — Search for bore shapes that meet target sonic properties (e.g. drone in D, toots in F, G, B). This is inspired by the work of [Frank Geipel](https://www.didgeridoo-physik.de/) (Computer-Aided Didgeridoo Sound Design, CADSD).

So the first part takes a geometry and predicts its sound; the second takes a target sound and searches for a matching geometry. To the best of our knowledge, DidgeLab is the first open toolkit that implements this inverse design.

The software is a **Python toolkit** rather than a point-and-click app: you use it from scripts or Jupyter notebooks. There is no graphical user interface. You need Python skills to use it.

## 2. Related works

- **Frank Geipel** — [Computer-Aided Didgeridoo Sound Design](https://www.didgeridoo-physik.de/) (CADSD). DidgeLab’s approach is based on our reading of his ideas and descriptions.
- **Dan Mapes-Riordan (1991)** — *Horn Modeling with Conical and Cylindrical Transmission Line Elements*. Foundation for the transmission-line acoustical model.
- **Andrea Ferroni ([YouTube](https://www.youtube.com/@AndreaFerroni))** — Didgeridoo acoustics and playing; e.g. DIDGMO explanation, backpressure, cylindrical vs conical bore.
- **Didgmo / Didjimp** — Existing tools for didgeridoo sound design; used here as a reference (DidgeLab’s impedance spectra match when given the same geometry).

## 3. Getting Started

### 3.1 Installation

**Prerequisites**

- Python 3.8+ (Conda or venv recommended)
- Optional: Cython and a C compiler for the fast simulation backend

**Use Pip Package**

```
pip install didgelab
```

### 3.2 Documentation

Here is an [overview of the available documentation](doc/README..nd). The best place to get started are the tutorials:

* [Didge Lab Tutorial - Part 1 - Introduction and acoustical simulation](doc/examples/tutorials/tutorial1.ipynb)
* [DidgeLab tutorial part 2 - Create geometries from parametric shapes](doc/examples/tutorials/tutorial2.ipynb)
* [DidgeLab tutorial part 3 - Optimization through Computational Evolution](doc/examples/tutorials/tutorial3.ipynb)

## 4. Developer Instructions

### 4.1 Compile from source

If you use the pip package, you do not need to compile it from source.

1. Clone the repository and go to the project directory:
   ```bash
   git clone https://github.com/jnehring/didge-lab/
   cd didge-lab
   ```
   (Or download and unzip the source.)

2. Create and activate a Conda environment:
   ```bash
   conda create -n didgelab python=3.8
   conda activate didgelab
   ```

3. Install the package (from the **repository root**; this builds the Cython extension and installs `didgelab`):
   ```bash
   pip install -e .
   ```
   Dependencies (NumPy, Cython, etc.) are installed as build/install dependencies. For a non-editable install: `pip install .`

4. Run a quick check (e.g. a script or notebook that runs a simulation and prints a small impedance table). If you see a frequency/impedance/note table, the installation is working.

### 4.2 Unit Tests (pytest)

From the **repository root**:

```bash
pytest
```

Tests live in `src/tests/`. To run a subset:

```bash
pytest src/tests/test_geo.py
pytest src/tests/test_KigaliShape.py -v
```

Some tests mock the Cython simulation backend so they run without building it; others require the full install (including the compiled `_cadsd` extension). Use the same Python environment in which you installed `didgelab`.

### 4.3 Building the Python package

- **Editable install (recommended for development)**  
  From the repo root:
  ```bash
  pip install -e .
  ```
  Changes in `src/didgelab` are picked up without reinstalling.

- **Regular install**  
  ```bash
  pip install .
  ```

- **Rebuild after changing Cython code**  
  Reinstall so the extension is recompiled:
  ```bash
  pip install -e . --no-build-isolation
  ```
  Or `pip install -e . --force-reinstall` if needed. A C compiler and Cython are required; see Cython docs for your OS.

- **Source distribution and wheel**  
  ```bash
  pip install build
  python -m build
  ```
  Outputs go to `dist/` (e.g. `pip install dist/didgelab-*.whl`).

- Tutorials in `examples/tutorials/` walk through acoustical simulation, shape parameters, and evolution.
- The `didgelab` package is documented with **module and function/class docstrings**; use `help()` in Python or read the source.

### 4.4 Build Documentation

**API documentation (pdoc)**  
Use [pdoc](https://pdoc.dev/) to generate API docs from docstrings. From the repo root, ensure the local package is used (e.g. after `pip install -e .` or with `PYTHONPATH=src`).

- **Development — live preview**  
  Serve docs locally and reload as you edit:
  ```bash
  pip install pdoc
  PYTHONPATH=src pdoc src/didgelab --http 8080
  ```
  Open http://localhost:8080. Stop with Ctrl+C.

- **Render static HTML**  
  Write the API docs into a folder (e.g. for deployment or offline use):
  ```bash
  pip install pdoc
  PYTHONPATH=src pdoc src/didgelab -o docs/html
  ```
  Then open `docs/html/index.html` in a browser or serve the `docs/html` directory with any static file server.

## 5. Source code overview

- **`didgelab`** — Main package:
  - `acoustical_simulation` — Entry point for running simulation (Python or Cython backend).
  - `geo` — Geometry class (segment list, load/save, scale, cones, volume, etc.).
  - `conv` — Note/frequency conversion (note names, cents, wavelengths).
  - `fft` — FFT and spectrum helpers (WAV, harmonics, fundamental, peaks).
  - `visualize` — Plot bore profiles (2D cross-section).
  - `app` — Global app, config, logging, publish/subscribe, services.
  - `sim` — Simulation: `sim_interface`, `tlm_python`, `tlm_cython` (TLM = transmission line model).
  - `evo` — Evolution: shapes, loss functions, mutators, evolution loop, checkpoint/loss logging.
- **Build** — From repo root: `pip install -e .` (see §4.1 and §4.3).

## 6. Licensing

DidgeLab is published under **Creative Commons BY-NC-SA 4.0**:

- **Share** — copy and redistribute in any medium or format  
- **Adapt** — remix, transform, and build upon the material  

Under these terms:

- **Attribution** — Give appropriate credit, link to the license, and indicate if changes were made.  
- **NonCommercial** — You may not use the material for commercial purposes.  
- **ShareAlike** — If you remix or build upon the material, you must distribute your contributions under the same license.  
- **No additional restrictions** — Do not apply legal or technical measures that restrict others from doing what the license permits.
