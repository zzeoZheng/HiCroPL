#!/bin/bash

#cd ../..

# custom config
DATA=/mnt/petrelfs/zhenghao/code/research/CasPL/DATA
TRAINER=HiCroPL

DATASET=$1
SEED=$2

if [ "$DATASET" = "imagenet" ]; then
    CFG=vit_b16_c2_ep40_batch128_16ctx
else
    CFG=vit_b16_c2_ep50_batch32_16ctx
fi
SHOTS=16
LOADEP=5
SUB=new


COMMON_DIR=${DATASET}/shots_${SHOTS}/${TRAINER}/${CFG}/seed${SEED}
MODEL_DIR=output/base2new/train_base/${COMMON_DIR}
DIR=output/base2new/test_${SUB}/${COMMON_DIR}
if [ -d "$DIR" ]; then
    echo "Evaluating model"
    echo "Results are available in ${DIR}. Resuming..."

    python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir ${DIR} \
    --model-dir ${MODEL_DIR} \
    --eval-only \
    DATASET.NUM_SHOTS ${SHOTS} \
    DATASET.SUBSAMPLE_CLASSES ${SUB}

else
    echo "Evaluating model"
    echo "Runing the first phase job and save the output to ${DIR}"

    python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir ${DIR} \
    --model-dir ${MODEL_DIR} \
    --eval-only \
    DATASET.NUM_SHOTS ${SHOTS} \
    DATASET.SUBSAMPLE_CLASSES ${SUB}
fi
