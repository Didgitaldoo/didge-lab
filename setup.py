"""
Build the Cython extension with NumPy include dirs.
Package metadata and package layout come from pyproject.toml.
"""
from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

_cadsd_ext = Extension(
    name="didgelab.sim.tlm_cython_lib._cadsd",
    sources=["src/didgelab/sim/tlm_cython_lib/_cadsd.pyx"],
    include_dirs=[np.get_include()],
    define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
)

setup(
    ext_modules=cythonize(
        [_cadsd_ext],
        compiler_directives={"language_level": "3"},
    ),
)
