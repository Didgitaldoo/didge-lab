from setuptools import setup, find_packagesp, Extension
from Cython.Build import cythonize
import numpy

setup(
    name='didgelab',
    version='2.0',    
    description='Didgeridoo computations',
    url='https://github.com/Didgitaldoo/didgelab',
    author='Jan Nehring',
    author_email='jan.nehring@outlook.com',
    license='Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International',
    include_package_data=True,
    packages=["didgelab", "didgelab.calc", "didgelab.calc.sim", "didgelab.evo", "didgelab.util"],
    setup_requires=[
        'setuptools>=18.0',  # Setuptools 18.0+ properly handles Cython extensions
        'cython==0.29.32',
    ],
    install_requires=['configargparse==1.5.3',
        'pandas==1.5.0',
        'scipy==1.10.1',
        'matplotlib==3.6.1',
        'prettytable==3.4.1',
        'seaborn==0.12.0',
        'tqdm==4.64.1',
        'cython==0.29.32',
        'flask==2.3.2',
        'jsonlines==4.0.0',
        'setuptools>=18.0',  # Setuptools 18.0+ properly handles Cython extensions
        ]
    ,
    classifiers=[
        'Programming Language :: Python :: 3.8',
    ],
    ext_modules = cythonize("didgelab/calc/sim/_cadsd.pyx"),
    include_dirs=[numpy.get_include()]
)
