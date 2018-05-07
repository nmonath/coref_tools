#!/usr/bin/env bash
#
#SBATCH --partition=longq    # Partition to submit to
#
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=24
#SBATCH --time=1-00:00         # Runtime in D-HH:MM
#SBATCH --mem-per-cpu=4000    # Memory in MB per cpu allocated

set -exu

config=$1
$PYTHON_EXEC -m coref.train.pw.CreateTrainingBatcher $config
