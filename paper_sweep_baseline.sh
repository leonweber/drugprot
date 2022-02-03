#!/usr/bin/env bash

RUN_DIR=paper_sweep/baseline/
SEEDS=( 1 2 3 42 105 )


for seed in "${SEEDS[@]}"
do
  if [ -f "$RUN_DIR/$seed/predictions.bioc.xml" ]
  then
    echo Skipping "$RUN_DIR/$seed"
  else
    CUDA_VISIBLE_DEVICES=0 python -m drugprot.train_multitask trainer=gpu hydra.run.dir="$RUN_DIR/$seed" seed="$seed"
  fi
done
