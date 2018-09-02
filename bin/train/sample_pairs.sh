#!/usr/bin/env bash

set -exu

config=$1

python -m grinch.train.sample_pairs $config