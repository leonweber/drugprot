#!/bin/bash

N_WORKER=20

for ((worker=0; worker<N_WORKER; worker++))
do
   pedl build_training_set --triples data/ctd_pedl.tsv --out data/ctd_pubtator.tsv --out_blinded data/ctd_pubtator_blinded.tsv --pubtator /glusterfs/dfs-gfs-dist/wbi-shared/corpora/pubtator-central-2021/output/BioCXML/  --n_worker $N_WORKER --worker_id "$worker" &
done
