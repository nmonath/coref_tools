#!/usr/bin/env bash


set -exu

in_tsv=$1
out_json=$2

python -m grinch.process.convert_wikilinks_mentions $in_tsv $out_json