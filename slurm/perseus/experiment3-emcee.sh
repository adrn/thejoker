#!/bin/bash
#SBATCH -J exp3-emcee           # job name
#SBATCH -o exp3-emcee.o%j             # output file name (%j expands to jobID)
#SBATCH -e exp3-emcee.e%j             # error file name (%j expands to jobID)
#SBATCH -n 130                   # total number of mpi tasks requested
#SBATCH -t 02:00:00             # run time (hh:mm:ss)
#SBATCH --mail-user=adrn@princeton.edu
#SBATCH --mail-type=begin       # email me when the job starts
#SBATCH --mail-type=end         # email me when the job finishes

cd /tigress/adrianp/projects/thejoker/scripts/

module load openmpi/gcc/1.10.2/64

source activate thejoker

export NSTEPS=65536
export SEED=42

# Run emcee on output from experiment 3
srun python continue-with-emcee.py -v --mpi -o \
--nsteps=$NSTEPS \
-f ../data/experiment3.h5 \
--hdf5-key="11" \
--name="experiment3-11.hdf5" \
--seed=$SEED
