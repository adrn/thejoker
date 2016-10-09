#!/bin/bash
#SBATCH -J exp2           # job name
#SBATCH -o exp2.o%j             # output file name (%j expands to jobID)
#SBATCH -e exp2.e%j             # error file name (%j expands to jobID)
#SBATCH -n 256                   # total number of mpi tasks requested
#SBATCH -t 00:30:00             # run time (hh:mm:ss) - 1.5 hours
#SBATCH --mail-user=adrn@princeton.edu
#SBATCH --mail-type=begin       # email me when the job starts
#SBATCH --mail-type=end         # email me when the job finishes

cd /tigress/adrianp/projects/thejoker/scripts/

module load openmpi/gcc/1.10.2/64

source activate thejoker

# Run experiment 1!
python make-experiment2-data.py -s 1234

srun python run-sampler.py -v --mpi -o \
-n 2**28 -s 42 \
-f ../data/experiment1.h5 \
--name="experiment2-fixed-jitter.h5" \
--fixed-jitter='0 m/s'

srun python run-sampler.py -v --mpi -o \
-n 2**28 -s 42 \
-f ../data/experiment1.h5 \
--name="experiment2-sample-jitter.h5" \
--log-jitter2-mean=10.5 --log-jitter2-std=0.25
