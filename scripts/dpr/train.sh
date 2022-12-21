#!/bin/bash

TRAIN_DATAPATH="./data/10k/nq-train.json"
VAL_DATAPATH="./data/10k/nq-dev.json"
TEST_DATAPATH="./data/10k/nq-dev.json"

CUDA_VISIBLE_DEVICES=2 python3 src/train.py \
    model=t5_encoder_model                      \
    trainer=gpu                                 \
    datamodule=drscl                            \
    datamodule.train_datapath=${TRAIN_DATAPATH} \
    datamodule.val_datapath=${VAL_DATAPATH}     \
    datamodule.test_datapath=${TEST_DATAPATH}   \
    logger=wandb                                \
    trainer.max_epochs=50                       \
    logger.wandb.project=dpr_test