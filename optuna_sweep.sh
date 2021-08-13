CUDA_VISIBLE_DEVICES=3 python -m drugprot.train_multitask -m trainer=gpu data=drugprot \
finetune_trainer=none hydra/launcher=ray +hydra.launcher.ray.remote.num_gpus=1 \
seed=42,73,105,200 \
model.use_none_class=false \
model.entity_side_information=ctd_entities.tsv,uniprot_entities.tsv,ctd_uniprot_entities.tsv,null \
model.pair_side_information=ctd_relations.tsv,null \
model.max_length=312 \
+trainer.accumulate_grad_batches=2 \
batch_size=16 \
model.use_doc_context=true
