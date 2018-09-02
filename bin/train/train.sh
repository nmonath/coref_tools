#!/usr/bin/env bash

set -exu

config=$1

python -m grinch.train.train_hac $config