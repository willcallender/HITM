#!/bin/bash


#SBATCH --job-name=HITM_Sims
#SBATCH --mail-type=ALL
#SBATCH -n 16
#SBATCH -p genacc_q
#SBATCH -t 14-00:00:00


module load anaconda/3.8.3


conda activate hitm
python /gpfs/home/wac20/hitm/main.py
