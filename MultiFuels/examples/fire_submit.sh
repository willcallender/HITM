#!/bin/bash


#SBATCH --job-name=batch_fireline
#SBATCH --mail-type=ALL
#SBATCH -a [01-100]
#SBATCH -p genacc_q
#SBATCH -t 14-00:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --mem=4G


module load matlab

/gpfs/home/wac20/MultiFuels/examples/fireLine_batch.sh