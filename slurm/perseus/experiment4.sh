#!/bin/bash
#SBATCH -J exp4           # job name
#SBATCH -o exp4.o%j             # output file name (%j expands to jobID)
#SBATCH -e exp4.e%j             # error file name (%j expands to jobID)
#SBATCH -n 260                   # total number of mpi tasks requested
#SBATCH -t 00:26:00             # run time (hh:mm:ss) - 1.5 hours
#SBATCH --mail-user=adrn@princeton.edu
#SBATCH --mail-type=begin       # email me when the job starts
#SBATCH --mail-type=end         # email me when the job finishes

cd /tigress/adrianp/projects/thejoker/scripts/

module load openmpi/gcc/1.10.2/64

source activate thejoker

# Run experiment 4!
srun python run-sampler.py -v --mpi -o \
-n 2**28 -s 42 \
-f ../data/troup-allVisit.h5 \
--hdf5-key="2M00110648+6609349" \
--name="experiment4-fixed-jitter.h5" \
--fixed-jitter='0 m/s'

srun python run-sampler.py -v --mpi -o \
-n 2**28 -s 42 \
-f ../data/troup-allVisit.h5 \
--hdf5-key="2M00110648+6609349" \
--name="experiment4-sample-jitter.h5" \
--log-jitter2-mean=10.5 --log-jitter2-std=1.
