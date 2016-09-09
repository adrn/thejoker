#!/bin/bash
#SBATCH -J exp2           # job name
#SBATCH -o exp2.o%j             # output file name (%j expands to jobID)
#SBATCH -e exp2.e%j             # error file name (%j expands to jobID)
#SBATCH -n 256                   # total number of mpi tasks requested
#SBATCH -t 01:45:00             # run time (hh:mm:ss) - 1.5 hours
#SBATCH --mail-user=adrn@princeton.edu
#SBATCH --mail-type=begin       # email me when the job starts
#SBATCH --mail-type=end         # email me when the job finishes

cd /tigress/adrianp/projects/thejoker/scripts/

module load openmpi/gcc/1.10.2/64

source activate thejoker

# Run experiment 2, dropping data points in batches of 4
srun python run-sampler.py -v --mpi -o \
-n 2**27 \
-f ../data/experiment2.h5 \
--hdf5-key="2M03080601+7950502-0"

srun python run-sampler.py -v --mpi -o \
-n 2**27 \
-f ../data/experiment2.h5 \
--hdf5-key="2M03080601+7950502-4"

srun python run-sampler.py -v --mpi -o \
-n 2**27 \
-f ../data/experiment2.h5 \
--hdf5-key="2M03080601+7950502-8"

srun python run-sampler.py -v --mpi -o \
-n 2**27 \
-f ../data/experiment2.h5 \
--hdf5-key="2M03080601+7950502-12"

srun python run-sampler.py -v --mpi -o \
-n 2**27 \
-f ../data/experiment2.h5 \
--hdf5-key="2M03080601+7950502-16"

srun python run-sampler.py -v --mpi -o \
-n 2**27 \
-f ../data/experiment2.h5 \
--hdf5-key="2M03080601+7950502-20"

srun python run-sampler.py -v --mpi -o \
-n 2**27 \
-f ../data/experiment2.h5 \
--hdf5-key="2M03080601+7950502-24"

srun python run-sampler.py -v --mpi -o \
-n 2**27 \
-f ../data/experiment2.h5 \
--hdf5-key="2M03080601+7950502-28"
