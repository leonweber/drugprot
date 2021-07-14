## Preparation of CTD database triplets
To prepare triplets from the CTD database run:
```bash
python prepare_ctd_triplets.py
```
This will download the necessary data files from the CTD website and store the prepared triplets 
as well as the entity and relation dictionary in the folder `data/ctd/dgl_ke/`.


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

