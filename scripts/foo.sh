#!/bin/bash
module load python3/intel/3.6.3
cd /home/$USER/justice
source /scratch/$USER/venv3/bin/activate
python3 -m pytest -s -m "not no_precommit" test "$@"
# python3 period.py
