#!/usr/bin/env bash

set -exu

input_file=$1
TIME=`(date +%Y-%m-%d-%H-%M-%S)`
vocab_file=$2
output_file=${3:-$input_file.$TIME.procesed.gz}

python -m grinch.process.preprocess_ments_tgx $input_file $output_file $vocab_file