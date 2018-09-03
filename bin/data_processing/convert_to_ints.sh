#!/usr/bin/env bash

set -exu

input_file=$1
TIME=`(date +%Y-%m-%d-%H-%M-%S)`
vocab_file=$2
output_file=${3:-$input_file.$TIME.ints.gz}

python -m grinch.process.convert_to_ints $input_file $output_file $vocab_file