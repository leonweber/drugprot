#!/usr/bin/env bash
RUN_DIR=paper_sweep/entity_markers/


CUDA_VISIBLE_DEVICES=2 python -m drugprot.train_multitask -m trainer=gpu \
model.use_cls=true model.use_starts=true,false model.use_ends=true,false \
hydra/launcher=ray +hydra.launcher.ray.remote.num_gpus=1 hydra.sweep.dir="$RUN_DIR/"

CUDA_VISIBLE_DEVICES=2 python -m drugprot.train_multitask -m trainer=gpu \
model.use_cls=true,false model.use_starts=true model.use_ends=true \
hydra/launcher=ray +hydra.launcher.ray.remote.num_gpus=1 hydra.sweep.dir="$RUN_DIR/"
