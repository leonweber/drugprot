#!/usr/bin/env bash

RUN_DIR=paper_sweep/transformer/


CUDA_VISIBLE_DEVICES=0,3 python -m drugprot.train_multitask -m trainer=gpu model.transformer=microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract,dmis-lab/biobert-v1.1,allenai/biomed_roberta_base,microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext hydra.sweep.dir="$RUN_DIR/" \
trainer.max_epochs=3,5,10 \
seed=1,2,3,42,105 \
data.dataset_to_batch_size.drugprot=8,16,32 \
hydra/launcher=ray +hydra.launcher.ray.remote.num_gpus=1
