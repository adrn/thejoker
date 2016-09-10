#!/bin/bash
#SBATCH -J exp1           # job name
#SBATCH -o exp1.o%j             # output file name (%j expands to jobID)
#SBATCH -e exp1.e%j             # error file name (%j expands to jobID)
#SBATCH -n 256                   # total number of mpi tasks requested
#SBATCH -t 00:30:00             # run time (hh:mm:ss) - 1.5 hours
#SBATCH --mail-user=adrn@princeton.edu
#SBATCH --mail-type=begin       # email me when the job starts
#SBATCH --mail-type=end         # email me when the job finishes

cd /tigress/adrianp/projects/thejoker/scripts/

module load openmpi/gcc/1.10.2/64

source activate thejoker

# Run experiment 1!
python make-experiment1-data.py -s 1988

srun python run-sampler.py -v --mpi -o \
-n 2**28 \
-f ../data/experiment1.h5 \
--name="experiment1.h5"
