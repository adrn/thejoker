#!/bin/bash
#SBATCH -J exp2           # job name
#SBATCH -o exp2.o%j             # output file name (%j expands to jobID)
#SBATCH -e exp2.e%j             # error file name (%j expands to jobID)
#SBATCH -n 512                   # total number of mpi tasks requested
#SBATCH -t 01:45:00             # run time (hh:mm:ss)
#SBATCH --mail-user=adrn@princeton.edu
#SBATCH --mail-type=begin       # email me when the job starts
#SBATCH --mail-type=end         # email me when the job finishes

cd /tigress/adrianp/projects/thejoker/scripts/

module load openmpi/gcc/1.10.2/64

source activate thejoker

python make-experiment2-data.py -s 42

export NSAMPLES="2**28"
export SEED=42

# Run experiment 2, inflating error-bars by factor of 1, 2, 4, 8
srun python run-sampler.py -v --mpi -o \
-n $NSAMPLES \
-f ../data/experiment2.h5 \
--name="experiment2-1.hdf5" \
--hdf5-key="1" \
--seed=$SEED

srun python run-sampler.py -v --mpi -o \
-n $NSAMPLES \
-f ../data/experiment2.h5 \
--name="experiment2-2.hdf5" \
--hdf5-key="2" \
--seed=$SEED

srun python run-sampler.py -v --mpi -o \
-n $NSAMPLES \
-f ../data/experiment2.h5 \
--name="experiment2-4.hdf5" \
--hdf5-key="4" \
--seed=$SEED

srun python run-sampler.py -v --mpi -o \
-n $NSAMPLES \
-f ../data/experiment2.h5 \
--name="experiment2-8.hdf5" \
--hdf5-key="8" \
--seed=$SEED
