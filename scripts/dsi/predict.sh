#!/bin/bash


TRAIN_DATAPATH="./data/nq_1k/nq-train.json"
VAL_DATAPATH="./data/nq_1k/nq-dev.json"
TEST_DATAPATH="./data/nq_1k/nq-test.json"
TRIE_PATH="./data/nq_1k/trie.pickle"

CKPT_PATH="./logs/train/runs/2022-12-20_23-58-25/checkpoints/epoch_021.ckpt"

CUDA_VISIBLE_DEVICES=0 python3 src/predict.py   \
    model=t5_model                              \
    trainer=gpu                                 \
    datamodule=drscl                            \
    task_name=predict                           \
    tags=['predict']                            \
    ckpt_path=$CKPT_PATH                        \
    datamodule.train_datapath=${TRAIN_DATAPATH} \
    datamodule.val_datapath=${VAL_DATAPATH}     \
    datamodule.test_datapath=${TEST_DATAPATH}   \
    model.net.trie_path=$TRIE_PATH