#!/usr/bin/env bash

set -exu

input=$1
threads=1
partition=${2:-titanx-short}
EMAIL=${3:-None}

dataset_name=`cat $CONFIG | jq -r .dataset_name`
model_name=`cat $CONFIG | jq -r .model_name`
alg_name=`cat $CONFIG | jq -r .alg_name`

TIME=`(date +%Y-%m-%d-%H-%M-%S)`

log_dir=logs/train/$dataset_name/$model_name/$alg_name/$TIME

export MKL_NUM_THREADS=$threads
export OPENBLAS_NUM_THREADS=$threads
export OMP_NUM_THREADS=$threads

mkdir -p $log_dir/

sbatch -J sample_pairs \
            -e $log_dir/sample_pairs.err \
            -o $log_dir/sample_pairs.log \
            --cpus-per-task $threads \
            --partition=$partition \
            --gres=gpu:1 \
            --ntasks=1 \
            --nodes=1 \
            --mem-per-cpu=12000 \
            --time=0-04:00 \
            --mail-user $USER@cs.umass.edu --mail-type=$EMAIL bin/train/train.sh $input
