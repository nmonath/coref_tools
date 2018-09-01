#!/usr/bin/env bash

set -exu

input=$1
corpus_name=$2
threads=$3
TIME=`(date +%Y-%m-%d-%H-%M-%S)`
partition=${4:-cpu}
EMAIL=${5:-None}

export MKL_NUM_THREADS=$threads
export OPENBLAS_NUM_THREADS=$threads
export OMP_NUM_THREADS=$threads

mkdir -p logs/train/ft/$corpus_name/${TIME}/
mkdir -p exp_out/$corpus_name/$TIME

sbatch -J ft-${corpus_name} \
            -e logs/train/ft/$corpus_name/${TIME}/${corpus_name}.err \
            -o logs/train/ft/$corpus_name/${TIME}/${corpus_name}.log \
            --cpus-per-task $threads \
            --partition=$partition \
            --ntasks=1 \
            --nodes=1 \
            --mem-per-cpu=6000 \
            --time=0-04:00 \
            --mail-user $USER@cs.umass.edu --mail-type=$EMAIL bin/train/fasttext/train_cbow_.sh $input exp_out/$corpus_name/$TIME/embeddings