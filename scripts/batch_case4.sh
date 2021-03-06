#!/bin/bash
#SBATCH -p long
#SBATCH -t 72:00:00
#SBATCH -N 1
#SBATCH -n 20
#SBATCH --mem-per-cpu=2048
#SBATCH --exclusive

export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export NUMEXPR_NUM_THREADS=2
ulimit -s unlimited
export KMP_STACKSIZE=2000m

# Load modules.
module purge
module load anaconda3

cd $SLURM_SUBMIT_DIR

cat args_case4.txt | xargs -n8 -P10 python eulerian_case4.py

wait
exit 0
