#!/usr/bin/env bash

set -exu

TIME=`(date +%Y-%m-%d-%H-%M-%S)`

input_file=$1
output_file=${2:-$input_file.$TIME.vocab}

python -m grinch.process.make_vocab $input_file $output_file