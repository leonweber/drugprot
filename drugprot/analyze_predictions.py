import argparse
import json
import random
import warnings
from collections import defaultdict, Counter
from pathlib import Path
from typing import Optional

import bioc
import numpy as np



def get_explanation_tokens(explanation, tokens, topk):
    explanation_tokens = []
    explanation = np.array(explanation)
    for i in np.argsort(explanation)[-topk:]:
        explanation_tokens.append(tokens[i])

    return explanation_tokens


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

    correct_explanations = defaultdict(list)
    confusion_explanations_false_class = defaultdict(list)
    confusion_explanations_true_class = defaultdict(list)
    confusion_count = defaultdict(int)
    confusion_texts = defaultdict(list)

    n_total = 0
    n_skipped = 0

    n_tps = 0
    n_fps = 0
    n_fns = 0


    for doc in collection.documents:
        rels = []
        anns = []

        for sentence in doc.passages[0].sentences:
            for ann in sentence.annotations:
                ann_type = ann.infons['type']
                start = ann.locations[0].offset
                end = start + ann.locations[0].length
                anns.append(f"{ann.id}\t{ann_type} {start} {end}\t{ann.text}\n")

            true_rels = [i for i in sentence.relations if "prob" not in i.infons and i.infons["type"].lower() != "none"]
            true_rel_set = {(i.get_node("head").refid, i.infons["type"], i.get_node("tail").refid) for i in true_rels}
            pred_rels = [i for i in sentence.relations if "prob" in i.infons]
            pred_rel_set = {(i.get_node("head").refid, i.infons["type"], i.get_node("tail").refid) for i in pred_rels if i.infons["type"].lower() != "none"}

            tps = true_rel_set & pred_rel_set
            fps = pred_rel_set - true_rel_set
            fns = true_rel_set - pred_rel_set

            n_tps += len(tps)
            n_fps += len(fps)
            n_fns += len(fns)

            pair_to_true_relations = defaultdict(set)
            pair_to_pred_relations = defaultdict(set)

            for rel in sentence.relations:
                head = rel.get_node("head").refid
                tail = rel.get_node("tail").refid
                if "prob" in rel.infons:
                    pair_to_pred_relations[(head, tail)].add(rel)
                else:
                    pair_to_true_relations[(head, tail)].add(rel)

            for rel in sentence.relations:
                n_total += 1
                head = rel.get_node("head").refid
                tail = rel.get_node("tail").refid
                rel_type = rel.infons["type"]
                if rel_type == "NONE":
                    continue
                signature = (head, rel_type, tail)
                if signature in tps:
                    if "prob" in rel.infons:
                        if "explanation" in rel.infons:
                            try:
                                explanation_tokens = get_explanation_tokens(explanation=json.loads(rel.infons["explanation"]),
                                                                            tokens=json.loads(rel.infons["tokens"].replace("'", '"')),
                                                                            topk=5)
                                correct_explanations[rel_type].extend(explanation_tokens)
                            except json.JSONDecodeError:
                                n_skipped += 1


                        continue # already written from true relations
                    suffix = ""
                elif signature in fps:
                    suffix = "_FP"
                    confusion_rels = pair_to_true_relations[(head, tail)]
                    if not confusion_rels:
                        confusion_rels = {"NONE"}
                    if "explanation" in rel.infons:
                        for confusion_rel in confusion_rels:
                            try:
                                confusion_type = confusion_rel.infons["type"]
                            except AttributeError:
                                confusion_type = confusion_rel
                            try:
                                explanation_tokens = get_explanation_tokens(explanation=json.loads(rel.infons["explanation"]),
                                                                            tokens=json.loads(rel.infons["tokens"].replace("'", '"')),
                                                                            topk=5)
                                confusion_explanations_false_class[(rel_type, confusion_type)].extend(explanation_tokens)
                                confusion_texts[(rel_type, confusion_type)].append(rel.infons["tokens"])
                                confusion_count[(rel_type, confusion_type)] += 1
                            except json.JSONDecodeError:
                                n_skipped += 1

                else:
                    suffix = "_FN"
                    confusion_rels = pair_to_pred_relations[(head, tail)]
                    for confusion_rel in confusion_rels:
                        if confusion_rel.infons["type"] != "NONE":
                            continue
                        if "explanation" in confusion_rel.infons:
                            try:
                                explanation_tokens = get_explanation_tokens(explanation=json.loads(confusion_rel.infons["explanation"]),
                                                                            tokens=json.loads(confusion_rel.infons["tokens"].replace("'", '"')),
                                                                            topk=5)
                            except json.JSONDecodeError:
                                n_skipped += 1
                                explanation_tokens = None
                        else:
                            explanation_tokens = None

                        assert confusion_rel.infons["type"] == "NONE"
                        if explanation_tokens:
                            confusion_explanations_false_class[(confusion_rel.infons["type"], rel_type)].extend(explanation_tokens)
                        confusion_count[(confusion_rel.infons["type"], rel_type)] += 1

                rel_type = rel.infons["type"] + suffix
                rels.append(f"R{len(rels) + 1}\t{rel_type} Arg1:{head} Arg2:{tail}\n")

        if args.brat_out:
            with open(str(args.brat_out / doc.id) + ".txt", "w") as f_txt, \
                    open(str(args.brat_out / doc.id) + ".ann", "w") as f_ann:
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
    warnings.warn(f"Skipped {n_skipped}/{n_total} explanations due to invalid tokens")
    print(f"F1: {n_tps/(n_tps + 0.5 * (n_fps + n_fns))}")

    for rel_type, explanations in correct_explanations.items():
        print("=============")
        print(rel_type)
        print("=============")
        print(Counter(explanations).most_common(10))
        print()
        print()

    for confusion_pair, n in sorted(confusion_count.items(), key=lambda x: x[1])[::-1]:
        print("=============")
        print(confusion_pair, n)
        print("=============")
        print(Counter(confusion_explanations_false_class[confusion_pair]).most_common(10))
        print()
        print()
