#!/bin/bash

TRAIN_DATAPATH="./data/nq_1k/nq-train.json"
VAL_DATAPATH="./data/nq_1k/nq-dev.json"
TEST_DATAPATH="./data/nq_1k/nq-test.json"
CKPT_PATH="./logs/train/runs/2022-12-20_21-45-20/checkpoints/epoch_022.ckpt"

CUDA_VISIBLE_DEVICES=0 python3 src/eval.py \
    model=t5_model                              \
    trainer=gpu                                 \
    datamodule=drscl                            \
    datamodule.train_datapath=${TRAIN_DATAPATH} \
    datamodule.val_datapath=${VAL_DATAPATH}     \
    datamodule.test_datapath=${TEST_DATAPATH}   \
    task_name=eval                              \
    tags=['test']                               \
    ckpt_path=$CKPT_PATH