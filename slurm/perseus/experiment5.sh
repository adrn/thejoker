#!/bin/bash
#SBATCH -J exp5           # job name
#SBATCH -o exp5.o%j             # output file name (%j expands to jobID)
#SBATCH -e exp5.e%j             # error file name (%j expands to jobID)
#SBATCH -n 256                   # total number of mpi tasks requested
#SBATCH -t 00:30:00             # run time (hh:mm:ss) - 1.5 hours
#SBATCH --mail-user=adrn@princeton.edu
#SBATCH --mail-type=begin       # email me when the job starts
#SBATCH --mail-type=end         # email me when the job finishes

cd /tigress/adrianp/projects/thejoker/scripts/

module load openmpi/gcc/1.10.2/64

source activate thejoker

# Run experiment 5!
python make-experiment15-data.py -s 1988

srun python run-sampler.py -v --mpi -o \
-n 2**28 -s 42 \
-f ../data/experiment5.h5 \
--name="experiment5.h5"
