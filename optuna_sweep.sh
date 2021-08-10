CUDA_VISIBLE_DEVICES=0,1,2 python -m drugprot.train -m model=multitask_with_distant trainer=gpu data=drugprot \
finetune_trainer=none hydra/launcher=ray +hydra.launcher.ray.remote.num_gpus=1 \
model.transformer=/vol/fob-vol1/mi15/weberple/glusterfs/models/RoBERTa-large-PM-M3-Voc/RoBERTa-large-PM-M3-Voc-hf,microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract,microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext,allenai/biomed_roberta_base,dmis-lab/biobert-v1.1 \
seed=1,2,3,4,5,6,7,8,9,42 \
model.lr=3e-5,5-e5,3e-4,5e-4,1e-5,5e-6 \
trainer.max_epochs=3,5,10 \
model.blind_entities=true,false