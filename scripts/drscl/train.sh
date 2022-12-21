#!/bin/bash

TRAIN_DATAPATH="./data/nq_1k/nq-train.json"
VAL_DATAPATH="./data/nq_1k/nq-dev.json"
TEST_DATAPATH="./data/nq_1k/nq-test.json"

CUDA_VISIBLE_DEVICES=0 python3 src/train.py \
    model=drscl_model                           \
    trainer=gpu                                 \
    datamodule=drscl                            \
    datamodule.train_datapath=${TRAIN_DATAPATH} \
    datamodule.val_datapath=${VAL_DATAPATH}     \
    datamodule.test_datapath=${TEST_DATAPATH}   \
    logger=wandb                                \
    trainer.max_epochs=50                       \
    logger.wandb.project=drscl_test