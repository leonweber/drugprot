import argparse
from collections import defaultdict
from pathlib import Path
from typing import Optional

import bioc
import numpy as np
from sklearn.metrics import classification_report, multilabel_confusion_matrix

from drugprot.models.entity_marker_baseline import LABEL_TO_ID, ID_TO_LABEL

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--brat_out", type=Path, default=None)
    args = parser.parse_args()

    with args.input.open() as f:
        collection = bioc.load(f)

    if args.brat_out:
        args.brat_out: Optional[Path]
        args.brat_out.mkdir(exist_ok=True)

    y_true = []
    y_pred = []

    for doc in collection.documents:
        rels = []
        anns = []

        all_bioc_rels = doc.passages[0].relations.copy()

        for sentence in doc.passages[0].sentences:
            for ann in sentence.annotations:
                ann_type = ann.infons['type']
                start = ann.locations[0].offset
                end = start + ann.locations[0].length
                anns.append(f"{ann.id}\t{ann_type} {start} {end}\t{ann.text}\n")

            all_bioc_rels.extend(sentence.relations)

        true_rels = [i for i in all_bioc_rels if "prob" not in i.infons]
        true_rel_set = {(i.get_node("head").refid, i.infons["type"], i.get_node("tail").refid) for i in true_rels}
        pred_rels = [i for i in all_bioc_rels if "prob" in i.infons]
        pred_rel_set = {(i.get_node("head").refid, i.infons["type"], i.get_node("tail").refid) for i in pred_rels}

        tps = true_rel_set & pred_rel_set
        fps = pred_rel_set - true_rel_set
        fns = true_rel_set - pred_rel_set

        pair_to_true_relations = defaultdict(set)
        pair_to_pred_relations = defaultdict(set)

        for rel in all_bioc_rels:
            head = rel.get_node("head").refid
            tail = rel.get_node("tail").refid
            if "prob" in rel.infons:
                pair_to_pred_relations[(head, tail)].add(rel.infons["type"])
            else:
                pair_to_true_relations[(head, tail)].add(rel.infons["type"])

            signature = (head, rel.infons["type"], tail)
            if signature in tps:
                if "prob" in rel.infons:
                    continue # already written from true relations
                suffix = ""
            elif signature in fps:
                suffix = "_FP"
            else:
                suffix = "_FN"
            rel_type = rel.infons["type"] + suffix
            rels.append(f"R{len(rels) + 1}\t{rel_type} Arg1:{head} Arg2:{tail}\n")

        for pair in set(pair_to_true_relations) | set(pair_to_pred_relations):
            true = np.zeros(len(LABEL_TO_ID))
            pred = np.zeros(len(LABEL_TO_ID))

            for rel in pair_to_true_relations[pair]:
                true[LABEL_TO_ID[rel]] = 1
            for rel in pair_to_pred_relations[pair]:
                pred[LABEL_TO_ID[rel]] = 1

            y_true.append(true)
            y_pred.append(pred)

        if args.brat_out:
            with (args.brat_out / doc.id).with_suffix(".txt").open("w") as f_txt, \
                    (args.brat_out / doc.id).with_suffix(".ann").open("w") as f_ann:
                f_txt.write(doc.passages[0].text)
                f_ann.writelines(anns)
                f_ann.writelines(rels)
            with (args.brat_out / "annotation.conf").open("w") as f:
                f.write(
                    """
[entities]	 
CHEMICAL
GENE-Y
GENE-N

[relations]
ACTIVATOR Arg1:CHEMICAL, Arg2:GENE-Y|GENE-N|GENE
AGONIST Arg1:CHEMICAL, Arg2:GENE-Y|GENE-N|GENE
AGONIST-ACTIVATOR Arg1:CHEMICAL, Arg2:GENE-Y|GENE-N|GENE
AGONIST-INHIBITOR Arg1:CHEMICAL, Arg2:GENE-Y|GENE-N|GENE
ANTAGONIST Arg1:CHEMICAL, Arg2:GENE-Y|GENE-N|GENE
DIRECT-REGULATOR Arg1:CHEMICAL, Arg2:GENE-Y|GENE-N|GENE
INDIRECT-DOWNREGULATOR Arg1:CHEMICAL, Arg2:GENE-Y|GENE-N|GENE
INDIRECT-UPREGULATOR Arg1:CHEMICAL, Arg2:GENE-Y|GENE-N|GENE
INHIBITOR Arg1:CHEMICAL, Arg2:GENE-Y|GENE-N|GENE
PART-OF Arg1:CHEMICAL, Arg2:GENE-Y|GENE-N|GENE
PRODUCT-OF Arg1:CHEMICAL, Arg2:GENE-Y|GENE-N|GENE
SUBSTRATE Arg1:CHEMICAL, Arg2:GENE-Y|GENE-N|GENE
SUBSTRATE_PRODUCT-OF Arg1:CHEMICAL, Arg2:GENE-Y|GENE-N|GENE

[events]

[attributes]
                    """
                )

    target_names = sorted(LABEL_TO_ID, key=lambda x: LABEL_TO_ID[x])
    print(classification_report(y_true=y_true, y_pred=y_pred, target_names=target_names))
