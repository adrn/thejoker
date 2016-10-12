#!/bin/bash
#SBATCH -J exp3           # job name
#SBATCH -o exp3.o%j             # output file name (%j expands to jobID)
#SBATCH -e exp3.e%j             # error file name (%j expands to jobID)
#SBATCH -n 260                   # total number of mpi tasks requested
#SBATCH -t 00:50:00             # run time (hh:mm:ss)
#SBATCH --mail-user=adrn@princeton.edu
#SBATCH --mail-type=begin       # email me when the job starts
#SBATCH --mail-type=end         # email me when the job finishes

cd /tigress/adrianp/projects/thejoker/scripts/

module load openmpi/gcc/1.10.2/64

source activate thejoker

python make-experiment3-data.py -s 42

export NSAMPLES="2**28"
export SEED=42

srun python run-sampler.py -v --mpi -o \
-n $NSAMPLES \
-f ../data/experiment3.h5 \
--name="experiment3-16.hdf5" \
--hdf5-key="16" \
--seed=$SEED \
--fixed-jitter='0 m/s'

srun python run-sampler.py -v --mpi -o \
-n $NSAMPLES \
-f ../data/experiment3.h5 \
--name="experiment3-8.hdf5" \
--hdf5-key="8" \
--seed=$SEED \
--fixed-jitter='0 m/s'

srun python run-sampler.py -v --mpi -o \
-n $NSAMPLES \
-f ../data/experiment3.h5 \
--name="experiment3-4.hdf5" \
--hdf5-key="4" \
--seed=$SEED \
--fixed-jitter='0 m/s'

srun python run-sampler.py -v --mpi -o \
-n $NSAMPLES \
-f ../data/experiment3.h5 \
--name="experiment3-2.hdf5" \
--hdf5-key="2" \
--seed=$SEED \
--fixed-jitter='0 m/s'
