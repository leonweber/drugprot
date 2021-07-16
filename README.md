## Run baseline model
```bash
(drugprot) guppi5 weberple 40 ( drugprot ) $ python -m drugprot.train data.train=null data.checkpoint="/glusterfs/dfs-gfs-dist/weberple-pub/drugprot/entity_marker_baseline.ckpt"
F1: 0.78: 100%|██████████████████████████████████████████████████████████████████████████████████| 750/750 [00:34<00:00, 21.45it/s]
[2021-07-09 10:57:00,330][__main__][INFO] - Wrote predictions to /vol/fob-wbib-vol2/wbi/weberple/projects/drugprot/logs/runs/2021-07-09/10-56-11/predictions.bioc.xml

(drugprot) guppi5 weberple 30 ( drugprot ) $ python analyze_predictions.py /vol/fob-wbib-vol2/wbi/weberple/projects/drugprot/logs/runs/2021-07-09/10-56-11/predictions.bioc.xml --brat_out analysis
Global seed set to 42
                        precision    recall  f1-score   support

                  NONE       0.00      0.00      0.00         0
             ACTIVATOR       0.79      0.72      0.75       246
               AGONIST       0.83      0.73      0.78       131
     AGONIST-ACTIVATOR       0.00      0.00      0.00        10
     AGONIST-INHIBITOR       0.00      0.00      0.00         2
            ANTAGONIST       0.92      0.91      0.92       218
      DIRECT-REGULATOR       0.69      0.63      0.66       457
INDIRECT-DOWNREGULATOR       0.79      0.79      0.79       332
  INDIRECT-UPREGULATOR       0.83      0.74      0.78       302
             INHIBITOR       0.85      0.88      0.87      1150
               PART-OF       0.73      0.77      0.75       257
            PRODUCT-OF       0.69      0.65      0.67       158
             SUBSTRATE       0.71      0.66      0.68       494
  SUBSTRATE_PRODUCT-OF       0.00      0.00      0.00         3

             micro avg       0.79      0.77      0.78      3760
             macro avg       0.56      0.54      0.55      3760
          weighted avg       0.79      0.77      0.78      3760
           samples avg       0.66      0.66      0.66      3760
```
The brat files for error analysis can then be found in `analysis`

## Entity pair classification
To (re-) generate the classification training and development files run:
```bash
python -m drugprot.bioc_to_cl_data \
  --input_bioc data/drugprot_biosyn_norm/train.bioc.xml \
  --output data/drugprot_entity/train.tsv
  
python -m drugprot.bioc_to_cl_data \
  --input_bioc data/drugprot_biosyn_norm/dev.bioc.xml \
  --output data/drugprot_entity/dev.tsv
```

To train the baseline entity classification model run:
```bash

```
