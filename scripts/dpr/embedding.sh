#!/bin/bash

CKPT_PATH="./logs/train/runs/2022-12-13_00-53-16/checkpoints/epoch_049.ckpt"

CUDA_VISIBLE_DEVICES=2 python3 src/embedding.py \
    model=t5_encoder_model          \
    trainer=gpu                     \
    datamodule=drscl                \
    task_name=embedding             \
    tags=['embedding']              \
    ckpt_path=$CKPT_PATH            