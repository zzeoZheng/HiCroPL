#!/bin/bash


# custom config
DATA=/mnt/petrelfs/zhenghao/code/research/CasPL/DATA
TRAINER=HiCroPL

DATASET=$1
SEED=$2

CFG=vit_b16_c2_ep5_batch32_2ctx_cross_datasets
SHOTS=16


DIR=output/${DATASET}/${TRAINER}/${CFG}_${SHOTS}shots/seed${SEED}
if [ -d "$DIR" ]; then
    echo "Results are available in ${DIR}."
else
    echo "Run this job and save the output to ${DIR}"

    srun -p mineru4s --gres=gpu:1 --cpus-per-task=10 --job-name=HiCroPL_training -u python train.py \
    --root ${DATA} \
    --seed ${SEED} \
    --trainer ${TRAINER} \
    --dataset-config-file configs/datasets/${DATASET}.yaml \
    --config-file configs/trainers/${TRAINER}/${CFG}.yaml \
    --output-dir ${DIR} \
    DATASET.NUM_SHOTS ${SHOTS}
fi