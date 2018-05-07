#!/usr/bin/env bash
#
#SBATCH --partition=longq    # Partition to submit to
#
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --time=0-04:00         # Runtime in D-HH:MM
#SBATCH --mem-per-cpu=4000    # Memory in MB per cpu allocated

set -exu

config=$1
outbase=${2-""}
dataname=${3:-""}
THREADS_PER_JOB=${4:-1}

export MKL_NUM_THREADS=$THREADS_PER_JOB
export OPENBLAS_NUM_THREADS=$THREADS_PER_JOB
export OMP_NUM_THREADS=$THREADS_PER_JOB

if [ -z $outbase ]
then
      $PYTHON_EXEC -m coref.train.pw.run.train_bin $config
else
      $PYTHON_EXEC -m coref.train.pw.run.train_bin $config --outbase $outbase --dataname $dataname
fi
