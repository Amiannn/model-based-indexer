#!/bin/bash

CKPT_PATH="./logs/train/runs/2022-12-13_00-53-16/checkpoints/epoch_049.ckpt"
EMBEDDING_PATH="./logs/embedding/runs/2022-12-14_01-44-57/embedding.pickle"

TEST_DATAPATH="./data/10k/nq-dev.tsv"

CUDA_VISIBLE_DEVICES=2 python3 src/predict.py \
    model=t5_encoder_model                       \
    trainer=gpu                                  \
    datamodule=drscl                             \
    task_name=predict                            \
    tags=['predict']                             \
    ckpt_path=$CKPT_PATH                         \
    datamodule.train_datapath=${TRAIN_DATAPATH}  \
    datamodule.val_datapath=${VAL_DATAPATH}      \
    datamodule.test_datapath=${TEST_DATAPATH}    \
    model.net.embedding_path=$EMBEDDING_PATH