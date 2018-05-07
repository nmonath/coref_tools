#!/usr/bin/env bash

set -exu

RUN_DIR=$1
CANOPIES_FILE=$2
CANOPY_PATH=$3
NUM_THREADS=${4:-24}
RANDOM_SEED=${5:-33}

num_runs=1
num_shufflings=10

TIME=`(date +%Y-%m-%d-%H-%M-%S)`

# Shuffle
# sh $COREF_ROOT/bin/util/shuffle_dataset.sh $dataset_file $num_runs

CANOPIES=($(cat $CANOPIES_FILE))
for i in `seq 1  $num_runs`
do
    for f in `ls -d ${RUN_DIR}/run*`
    do
        CONFIG=$f/config.json
        MODEL_NAME=`cat $CONFIG | jq -r .clustering_scheme`
        RUN=`basename $f`
        export LOG_LATEST=`pwd`/logs/inf/${MODEL_NAME}/${TIME}

        for shuffling in `seq 1  $num_shufflings`
        do

            SHUFF_DIR="shuffling_${shuffling}"
            mkdir -p logs/inf/${MODEL_NAME}/${TIME}/${RUN}/${SHUFF_DIR}

            OUTBASE="$TIME/${RUN}/$SHUFF_DIR"
            for CANOPY in "${CANOPIES[@]}"
            do
                MENTS_FILE="${CANOPY_PATH}/${CANOPY}/ments.json"
                POINTS_FILE="${CANOPY_PATH}/${CANOPY}/eval_pts.tsv"
                TEST_FILE=$MENTS_FILE

                sh $COREF_ROOT/bin/inf/hac/approx_inf_hac_profile.sh \
                    $CONFIG $TEST_FILE $NUM_THREADS $OUTBASE  ${CANOPY} ${POINTS_FILE} ${RANDOM_SEED} \
                     > logs/inf/${MODEL_NAME}/${TIME}/${RUN}/$SHUFF_DIR/${CANOPY}.log
            done
            RANDOM_SEED=$(( RANDOM_SEED + 1 ))
        done
    done
done
exit