#!/usr/bin/env bash

CUDA_VISIBLE_DEVICES=1 python -m drugprot.run_generative \
    --model_name_or_path razent/SciFive-large-Pubmed \
    --do_train \
    --train_file data/drugprot/train_generative.csv \
    --output_dir logs/generative/0 \
    --overwrite_output_dir \
    --per_device_train_batch_size=8 \
    --text_column source \
    --summary_column target \
    --fp16
