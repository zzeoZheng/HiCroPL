#!/bin/bash

# custom config
DATA="/path/to/dataset/folder"
TRAINER=HiCroPL

DATASET=$1
CFG=vit_b16_c2_ep50_batch32_16ctx_few_shot
SHOTS=$2


DIR=output/${DATASET}/${TRAINER}/${CFG}_${SHOTS}shots/seed${SEED}
if [ -d "$DIR" ]; then
    echo " The results exist at ${DIR}"
else
    echo "Run this job and save the output to ${DIR}"
    python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir ${DIR} \
    DATASET.NUM_SHOTS ${SHOTS}
fi

