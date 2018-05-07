#!/usr/bin/env bash

export COREF_ROOT=`pwd`

export PYTHON_EXEC=python
export PYTHONPATH=$COREF_ROOT/src/python:$PYTHONPATH

if [ ! -f $COREF_ROOT/.gitignore ]; then
    echo ".gitignore" > $COREF_ROOT/.gitignore
    echo "target" >> $COREF_ROOT/.gitignore
    echo ".idea" >> $COREF_ROOT/.gitignore
    echo "__pycache__" >> $COREF_ROOT/.gitignore
    echo "dep" >> $COREF_ROOT/.gitignore
    echo "data" >> $COREF_ROOT/.gitignore
    echo "test_out" >> $COREF_ROOT/.gitignore
    echo "experiments_out" >> $COREF_ROOT/.gitignore
    echo ".DS_STORE" >> $COREF_ROOT/.gitignore
    echo "*.iml" >> $COREF_ROOT/.gitignore
fi
