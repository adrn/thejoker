#!/bin/bash
#SBATCH -J sampler           # job name
#SBATCH -o sampler.o%j             # output file name (%j expands to jobID)
#SBATCH -e sampler.e%j             # error file name (%j expands to jobID)
#SBATCH -n 256                   # total number of mpi tasks requested
#SBATCH -p normal          # queue (partition) -- normal, development, etc.
#SBATCH -A TG-AST150023         # Project ID
#SBATCH -t 02:00:00             # run time (hh:mm:ss) - 1.5 hours
#SBATCH --mail-user=adrn@princeton.edu
#SBATCH --mail-type=begin       # email me when the job starts
#SBATCH --mail-type=end         # email me when the job finishes

cd $WORK/projects/thejoker/scripts/

source activate thejoker

ibrun python run-sampler.py -v --id='2M03080601+7950502' --mpi -n 2**24 -o
#ibrun python run-sampler.py -v --id='2M00110648+6609349' --mpi -n 2**24
#ibrun python run-sampler.py -v --id='2M00344509+8512058' --mpi -n 2**24
