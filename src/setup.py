"""
Setup script for didgelab. Builds the Cython extension during install.
Run from the directory containing this file (didge-lab-3/src):
  pip install .
  pip install -e .
"""
from setuptools import setup, find_packages, Extension
from Cython.Build import cythonize
import numpy as np

# Cython extension: compiled CADSD core used by didgelab.sim.tlm_cython
_cadsd_ext = Extension(
    name="didgelab.sim.tlm_cython_lib._cadsd",
    sources=["didgelab/sim/tlm_cython_lib/_cadsd.pyx"],
    include_dirs=[np.get_include()],
    define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
)

setup(
    name="didgelab",
    version="2.0",
    description="Acoustical simulation and computer-aided design of didgeridoos",
    long_description=open("README.md", encoding="utf-8").read() if __import__("os").path.exists("README.md") else "",
    long_description_content_type="text/markdown",
    url="https://github.com/didgitaldoo/didge-lab",
    author="Jan Nehring",
    author_email="jan.nehring@outlook.com",
    license="Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International",
    python_requires=">=3.8",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "configargparse>=1.5",
        "pandas>=1.5",
        "scipy>=1.10",
        "matplotlib>=3.6",
        "numpy",
        "seaborn>=0.12",
        "tqdm>=4.64",
    ],
    extras_require={
        "dev": ["pytest>=8", "Cython"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
    ],
    ext_modules=cythonize(
        [_cadsd_ext],
        compiler_directives={"language_level": "3"},
    ),
)
