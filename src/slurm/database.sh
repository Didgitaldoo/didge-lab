#!/bin/bash

'''
./usrun.sh -p RTXA6000 --mem 24GB --gpus 0 -c32 /netscratch/nehring/projects/music/didgelab2/src/slurm/database.sh
'''
export PIP_INDEX_URL=http://pypi-cache/index
export PIP_TRUSTED_HOST=pypi-cache
export PIP_NO_CACHE=true

cd "$(dirname "$0")"
cd ../

pip install -r requirements.txt

cd didgelab/calc/sim
python setup.py build_ext --inplace

cd ../../../

export NUM_CPUS=32

python -m didgelab.db.fill_db