#!/usr/bin/env bash
#

set -exu

LEVELS=$1
OUT=${2:-""}

if [ -z $OUT]
then
      $PYTHON_EXEC -m coref.util.create_hsep1d $LEVELS
else
      $PYTHON_EXEC -m coref.util.create_hsep1d $LEVELS --out $OUT
fi

