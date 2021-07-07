#!/bin/bash
#SBATCH -p normal
#SBATCH -t 48:00:00
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

cat args_case1.txt | xargs -n8 -P10 python eulerian_case1.py

wait
exit 0
