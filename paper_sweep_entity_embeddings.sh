#!/usr/bin/env bash

RUN_DIR=paper_sweep/entity_embeddings/
VALS=( /glusterfs/dfs-gfs-dist/saengema-pub/drugprot/embeddings/ComplEx_800 /glusterfs/dfs-gfs-dist/saengema-pub/drugprot/embeddings/DistMult_full_200 /glusterfs/dfs-gfs-dist/saengema-pub/drugprot/embeddings/pubtator_embeddings /glusterfs/dfs-gfs-dist/saengema-pub/drugprot/embeddings/RESCAL_full_200 )
SEEDS=( 1 2 3 42 105 )

for seed in "${SEEDS[@]}"
do
  for val in "${VALS[@]}"
  do
    if [ -f "$RUN_DIR/$val/$seed/predictions.bioc.xml" ]
    then
      echo Skipping "$RUN_DIR/$val/$seed"
    else
      CUDA_VISIBLE_DEVICES=3 python -m drugprot.train_multitask trainer=gpu model.entity_embeddings="$val" hydra.run.dir="$RUN_DIR/$val/$seed" seed="$seed"
    fi
  done
done
