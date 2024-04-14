#!/bin/bash
# ./usrun.sh -p  batch --gpus 0 /netscratch/nehring/projects/music/didgelab2/src/slurm/init.sh

export PIP_INDEX_URL=http://pypi-cache/index
export PIP_TRUSTED_HOST=pypi-cache
export PIP_NO_CACHE=true

cd "$(dirname "$0")"
cd ../

pip install -r requirements.txt

cd didgelab/calc/sim
python setup.py build_ext --inplace

cd ../../../

python -m experiments.arusha.evolve_arusha