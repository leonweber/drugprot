#!/usr/bin/env bash

RUN_DIR=paper_sweep/backtranslation/
VALS=( drugprot_backtranslation drugprot_backtranslation_wmt )
SEEDS=( 1 2 3 42 105 )


for seed in "${SEEDS[@]}"
do
  for val in "${VALS[@]}"
    do
    if [ -f "$RUN_DIR/$val/$seed/predictions.bioc.xml" ]
    then
      echo Skipping "$RUN_DIR/$val/$seed"
    else
        CUDA_VISIBLE_DEVICES=1 python -m drugprot.train trainer=gpu data="$val" hydra.run.dir="$RUN_DIR/$val/$seed" seed="$seed"
    fi
    done
done
