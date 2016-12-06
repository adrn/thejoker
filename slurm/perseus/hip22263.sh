#!/bin/bash
#SBATCH -J hip22263           # job name
#SBATCH -o hip22263.o%j             # output file name (%j expands to jobID)
#SBATCH -e hip22263.e%j             # error file name (%j expands to jobID)
#SBATCH -n 256                   # total number of mpi tasks requested
#SBATCH -t 02:00:00             # run time (hh:mm:ss) - 1.5 hours
#SBATCH --mail-user=adrn@princeton.edu
#SBATCH --mail-type=begin       # email me when the job starts
#SBATCH --mail-type=end         # email me when the job finishes

cd /tigress/adrianp/projects/thejoker/scripts/

module load openmpi/gcc/1.10.2/64

source activate thejoker

# Exoplanets!
srun python run-sampler.py  -v --mpi -o \
-n 2**25 -s 42 \
-f "../data/HIP22263.hdf5" \
--log-jitter2-mean=2. --log-jitter2-std=4.

