#!/usr/bin/env bash
#
#SBATCH --partition=defq    # Partition to submit to
#
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --time=0-01:00         # Runtime in D-HH:MM
#SBATCH --mem-per-cpu=4000    # Memory in MB per cpu allocated
set -exu

config=$1
test_file=${2:-""}
num_threads=${3:-24}
outbase=${4:-""}
canopy_name=${5:-""}
points_file=${6:-""}

export MKL_NUM_THREADS=$num_threads
export OPENBLAS_NUM_THREADS=$num_threads
export OMP_NUM_THREADS=$num_threads

# Add in a parameter for the root dir of these runs.

if [ -z $test_file ]
then
      $PYTHON_EXEC -m coref.inf.hac.online_pwehac_sl $config
else
      $PYTHON_EXEC -m coref.inf.hac.online_pwehac_sl $config \
          --test_file $test_file --outbase $outbase --canopy_name ${canopy_name} --points_file ${points_file}
fi

