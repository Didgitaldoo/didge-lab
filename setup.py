"""
Build the Cython extension with NumPy include dirs.
Package metadata and package layout come from pyproject.toml.

Uses the pre-generated _cadsd.c to avoid Cython producing absolute paths
during build (which setuptools rejects). Regenerate with: cython src/didgelab/sim/tlm_cython_lib/_cadsd.pyx
"""
import os
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import numpy as np

_CADSD_C = "src/didgelab/sim/tlm_cython_lib/_cadsd.c"

_cadsd_ext = Extension(
    name="didgelab.sim.tlm_cython_lib._cadsd",
    sources=[_CADSD_C],
    include_dirs=[np.get_include()],
    define_macros=[
        ("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION"),
        # MSVC's <complex.h> exposes complex numbers as a struct (`_Dcomplex`)
        # rather than C99 `double _Complex`, which is incompatible with the
        # `__pyx_t_double_complex` typedef Cython generates. Forcing
        # CYTHON_CCOMPLEX=0 makes the generated .c use Cython's own
        # struct-based complex type for all arithmetic, which compiles cleanly
        # under MSVC and remains correct under gcc/clang.
        ("CYTHON_CCOMPLEX", "0"),
    ],
)


class BuildExt(build_ext):
    """Ensure get_source_files returns relative paths for manifest (setuptools rejects absolute)."""

    def get_source_files(self):
        return [_CADSD_C]


setup(
    ext_modules=[_cadsd_ext],
    cmdclass={"build_ext": BuildExt},
)
