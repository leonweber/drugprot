## Preparation of CTD database triplets
To prepare triplets from the CTD database run:
```bash
python -m drugprot.kge.prepare_ctd
```
This will download the necessary data files from the CTD website and store the prepared triplets 
as well as the entity and relation dictionary in the folder `data/ctd/dgl_ke/` resp. `data/ctd/dgl_ke_full/`.
The former will have a real train / validation split whereas the latter contains all triplets in the 
train set and a random sample as validation split.


## KG-Embedding Framework

For training knowledge graph embeddings the DGL-KE tool was used:
- Documentation:  https://dglke.dgl.ai/doc/index.html
- GitHub-Repo: https://github.com/awslabs/dgl-ke

See the documentation for installation instructions etc.

## Training of KG-Embeddings
- Train RotatE embeddings
```bash
- CUDA_VISIBLE_DEVICES=1 DGLBACKEND=pytorch dglke_train --model_name RotatE -de \
--data_path ../data/ctd/dgl_ke --dataset dgl_ke --format udd_hrt --data_files entities.dict relation.dict train.tsv valid.tsv valid.tsv \
--delimiter "$(echo -en '\t')" --batch_size 1000 --neg_sample_size 200 --hidden_dim 600 --gamma 19.9 --lr 0.25 --max_step 10000 --log_interval 100 \
--batch_size_eval 16 -adv --regularization_coef 1.00E-09 --num_thread 1 --num_proc 1 --valid --gpu 0 --test
```

- Train ComplEX embeddings
```bash
CUDA_VISIBLE_DEVICES=0 DGLBACKEND=pytorch dglke_train --model_name ComplEx \
--data_path ../data/ctd/dgl_ke --dataset dgl_ke --format udd_hrt --data_files entities.dict relation.dict train.tsv valid.tsv valid.tsv \
--delimiter "$(echo -en '\t')" --batch_size 1000 --neg_sample_size 200 --hidden_dim 400 --gamma 19.9 --lr 0.25 --max_step 10000 --log_interval 100 \
--batch_size_eval 16 -adv --regularization_coef 1.00E-09 --num_thread 1 --num_proc 1 --valid --gpu 0 --test
```

- Train DistMult embeddings
```bash
CUDA_VISIBLE_DEVICES=1 DGLBACKEND=pytorch dglke_train --model_name DistMult \
--data_path ../data/ctd/dgl_ke --dataset dgl_ke --format udd_hrt --data_files entities.dict relation.dict train.tsv valid.tsv valid.tsv \
--delimiter "$(echo -en '\t')" --batch_size 1000 --neg_sample_size 200 --hidden_dim 400 --gamma 19.9 --lr 0.25 --max_step 10000 --log_interval 100 \
--batch_size_eval 16 -adv --regularization_coef 1.00E-09 --num_thread 1 --num_proc 1 --valid --gpu 0 --test
```

- Train RESCAL embeddings
```bash
CUDA_VISIBLE_DEVICES=0 DGLBACKEND=pytorch dglke_train --model_name RESCAL \
--data_path ../data/ctd/dgl_ke --dataset dgl_ke --format udd_hrt --data_files entities.dict relation.dict train.tsv valid.tsv valid.tsv \
--delimiter "$(echo -en '\t')" --batch_size 1000 --neg_sample_size 200 --hidden_dim 400 --gamma 19.9 --lr 0.25 --max_step 10000 --log_interval 100 \
--batch_size_eval 16 -adv --regularization_coef 1.00E-09 --num_thread 1 --num_proc 1 --valid --gpu 0 --test
```

## Convert the DGL-KE embeddings
To convert the generated embeddings to torch tensors (and calculate and add the DRUG-UNK/GENE-UNK embeddings) run:
```bash
python -m drugprot.kge.convert_dglke_embs --emb_file ckpts/DistMult_dgl_ke_7/dgl_ke_DistMult_entity.npy \
  --data_dir data/ctd/dgl_ke --output_dir data/embeddings/dist_mult
```
