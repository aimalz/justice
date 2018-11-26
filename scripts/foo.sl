#!/bin/bash -l
#SBATCH --nodes 1
#SBATCH -J justice181121
#SBATCH --mail-user=aimalz@nyu.edu
#SBATCH --mail-type=ALL
#SBATCH -t 06:00:00
#SBATCH --ntasks-per-node 28
#run the application:
bash /home/aim267/justice/foo.sh
