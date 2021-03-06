CUDA_VISIBLE_DEVICES=0,1,2,3 python -m drugprot.train_multitask -m trainer=gpu data=drugprot \
finetune_trainer=none hydra/launcher=ray +hydra.launcher.ray.remote.num_gpus=1 \
seed=42,105 \
model.use_none_class=false \
model.entity_side_information=null \
model.pair_side_information=null \
model.entity_embeddings=/glusterfs/dfs-gfs-dist/saengema-pub/drugprot/embeddings/DistMult_full_200,/glusterfs/dfs-gfs-dist/saengema-pub/drugprot/embeddings/RESCAL_full_200,/glusterfs/dfs-gfs-dist/saengema-pub/drugprot/embeddings/pubtator_embeddings,null \
model.max_length=128 \
model.weight_decay=0.0,1e-2,1e-3 \
+trainer.accumulate_grad_batches=1 \
trainer.max_epochs=3,5,10 \
batch_size=32 \
model.use_doc_context=false

