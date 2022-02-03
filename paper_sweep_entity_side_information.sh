#!/usr/bin/env bash

RUN_DIR=paper_sweep/entity_side_information/
SEEDS=( 1 2 3 42 105 )
VALS=( ctd_entities.tsv uniprot_entities.tsv ctd_uniprot_entities.tsv /glusterfs/dfs-gfs-dist/saengema-pub/drugprot/texts/mesh_to_targets_desc.tsv /glusterfs/dfs-gfs-dist/saengema-pub/drugprot/texts/mesh_to_targets_short.tsv /glusterfs/dfs-gfs-dist/saengema-pub/drugprot/texts/mesh_to_pharmacodynamics.tsv )


for seed in "${SEEDS[@]}"
do
  for val in "${VALS[@]}"
  do
    if [ -f "$RUN_DIR/$val/$seed/predictions.bioc.xml" ]
    then
      echo Skipping "$RUN_DIR/$val/$seed"
    else
      CUDA_VISIBLE_DEVICES=2 python -m drugprot.train_multitask trainer=gpu model.entity_side_information="$val" hydra.run.dir="$RUN_DIR/$val/$seed" seed="$seed"
    fi
  done
done
