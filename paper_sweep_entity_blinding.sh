#!/usr/bin/env bash

RUN_DIR=paper_sweep/blind_entities/
VALS=( true )
SEEDS=( 1 2 3 42 105 )

for seed in "${SEEDS[@]}"
do
  for val in "${VALS[@]}"
  do
    if [ -f "$RUN_DIR/$val/$seed/predictions.bioc.xml" ]
    then
      echo Skipping "$RUN_DIR/$val/$seed"
    else
      CUDA_VISIBLE_DEVICES=0 python -m drugprot.train_multitask trainer=gpu model.blind_entities="$val" hydra.run.dir="$RUN_DIR/$val/$seed" seed="$seed"
    fi
  done
done
