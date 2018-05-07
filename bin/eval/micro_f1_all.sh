#!/usr/bin/env bash
#SBATCH --partition=defq    # Partition to submit to
#
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --time=1-00:00         # Runtime in D-HH:MM
#SBATCH --mem-per-cpu=1000    # Memory in MB per cpu allocated
#set -exu

RUN_DIR=$1
MODEL=${2:-"model"}
DATASET=${3:-"dataset"}

for f in `ls -d ${RUN_DIR}/run*`
do
    RUN=`basename $f`
    RESULTS=$f/results

    for MODEL_DIR in `ls -d $RESULTS/*`
    do
        echo $MODEL_DIR
        for DATE_DIR in `ls -d $MODEL_DIR/*`
        do
            INDIR=$DATE_DIR/$RUN
            for SHUF_DIR in `ls -d $INDIR/*`
            do
                SHUF_BASE=`basename $SHUF_DIR`
                if [[ $SHUF_BASE == shuffling_* ]]
                then
                    echo "SHUFF DIR: $SHUF_DIR"
                    if ! [ -f ${SHUF_DIR}/micro_f1_thresholded.tsv ]; then
                        find ${SHUF_DIR} -name tree.tsv -exec cat {} \; | grep -v 'None$' | cut -f 1,3  > ${SHUF_DIR}/gold_cluster_assignments.tsv
#                        find ${SHUF_DIR} -name tree.tsv.upperbound -exec cat {} \; > ${SHUF_DIR}/all_upperbound_predictions.tsv
                        find ${SHUF_DIR} -name thresholded.tsv -exec cat {} \; > ${SHUF_DIR}/thresholded_clustering.tsv

                        sh $XCLUSTER_ROOT/bin/util/score_pairwise.sh \
                                ${SHUF_DIR}/thresholded_clustering.tsv \
                        ${SHUF_DIR}/gold_cluster_assignments.tsv \
                        ${MODEL} ${DATASET} None true > ${SHUF_DIR}/micro_f1_thresholded.tsv
                    fi
                fi
            done
        done
    done
done
