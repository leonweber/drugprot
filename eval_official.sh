#!/usr/bin/env bash

python -m drugprot.bioc_preds_to_drugprot --input $1 --output tmp_preds.tsv;
python ext/drugprot-evaluation-library/main.py \
  -g data/drugprot-gs-training-development/development/drugprot_development_relations.tsv \
  -e data/drugprot-gs-training-development/development/drugprot_development_entities.tsv \
  --pmids data/drugprot-gs-training-development/development/drugprot_development_pmids.txt \
  -p tmp_preds.tsv