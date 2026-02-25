# Codes for acoustical simulation in Cython

Build the Cython extension from the **package root** (`src`), not from this directory:

```bash
cd path/to/didge-lab-3/src
python setup.py build_ext --inplace
```

Or install the package (builds the extension automatically):

```bash
cd path/to/didge-lab-3/src
pip install -e .
```

If you run `setup.py build_ext --inplace` from inside this directory, the copy step fails because the destination path is computed relative to the wrong place.