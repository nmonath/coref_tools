#!/usr/bin/env bash

set -exu

input=$1
output=$2


fasttext skipgram -input $input -output $output -minn 2 -maxn 5 -dim 100 -ws 10