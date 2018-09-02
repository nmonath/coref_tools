#!/usr/bin/env bash

set -exu

input_file=$1
TIME=`(date +%Y-%m-%d-%H-%M-%S)`
vocab_file=$2
output_file=${3:-$input_file.$TIME.featurized.gz}

python -m grinch.process.featurize_ments $input_file $output_file $vocab_file