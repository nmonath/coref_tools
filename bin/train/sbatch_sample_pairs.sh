#!/usr/bin/env bash

set -exu

input=$1
threads=1
partition=${2:-cpu}
EMAIL=${3:-None}

TIME=`(date +%Y-%m-%d-%H-%M-%S)`

export MKL_NUM_THREADS=$threads
export OPENBLAS_NUM_THREADS=$threads
export OMP_NUM_THREADS=$threads

mkdir -p logs/train/sample_pairs/${TIME}/

sbatch -J sample_pairs \
            -e logs/train/sample_pairs/${TIME}/sample_pairs.err \
            -o logs/train/sample_pairs/${TIME}/sample_pairs.log \
            --cpus-per-task $threads \
            --partition=$partition \
            --ntasks=1 \
            --nodes=1 \
            --mem-per-cpu=6000 \
            --time=0-04:00 \
            --mail-user $USER@cs.umass.edu --mail-type=$EMAIL bin/train/sample_pairs.sh $input
