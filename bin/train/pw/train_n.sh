#!/usr/bin/env bash

set -exu

config=$1
num_runs=$2
dataname=$3

TIME=`(date +%Y-%m-%d-%H-%M-%S)`
log_dir=/tmp/$TIME
mkdir -p $log_dir

for i in `seq 1 $num_runs`
do
      OUTBASE=${TIME}/run_${i}
      $PYTHON_EXEC -m coref.train.pw.run.train_bin $config --outbase $OUTBASE --dataname $dataname > $log_dir/${dataname}_run_$i.log &
done
