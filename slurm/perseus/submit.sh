#!/bin/bash
#SBATCH -J sampler           # job name
#SBATCH -o sampler.o%j             # output file name (%j expands to jobID)
#SBATCH -e sampler.e%j             # error file name (%j expands to jobID)
#SBATCH -n 256                   # total number of mpi tasks requested
#SBATCH -t 02:00:00             # run time (hh:mm:ss) - 1.5 hours
#SBATCH --mail-user=adrn@princeton.edu
#SBATCH --mail-type=begin       # email me when the job starts
#SBATCH --mail-type=end         # email me when the job finishes

cd /tigress/adrianp/projects/thejoker/scripts/

module load openmpi/gcc/1.10.2/64

source activate thejoker

#srun python run-sampler.py -v --id='2M03080601+7950502' --mpi -n 2**5 -o
#srun python run-sampler.py -v --id='2M00110648+6609349' --mpi -n 2**24
#srun python run-sampler.py -v --id='2M00344509+8512058' --mpi -n 2**24

# Exoplanets!
srun python run-sampler.py -v --mpi -n 2**24 -o -f "../data/HIP102152_result.h5"
