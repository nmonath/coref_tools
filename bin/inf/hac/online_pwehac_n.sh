#!/usr/bin/env bash

set -exu

RUNS_DIR=$1

for f in `ls -d $RUNS_DIR/run*`
do
      CONFIG=$f/config.json
      LOG=$f/inf.log
      $PYTHON_EXEC -m coref.inf.hac.online_pwehac_sl $CONFIG > $LOG &
done
exit