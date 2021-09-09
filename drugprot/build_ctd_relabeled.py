import json
from collections import defaultdict, Counter

import bioc
import pandas as pd

from drugprot.analyze_predictions import get_explanation_tokens


def get_base_explanations(documents):
    confusion_explanations = defaultdict(list)
    correct_explanations = defaultdict(list)
    for doc in documents:
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
                                                                            topk=10)
                                correct_explanations[rel_type].append(set(explanation_tokens))
                            except json.JSONDecodeError:
                                pass


                        continue # already written from true relations
                elif signature in fps:
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
                                confusion_explanations[(rel_type, confusion_type)].append(explanation_tokens)
                            except json.JSONDecodeError:
                                pass

                else:
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
                                explanation_tokens = None
                        else:
                            explanation_tokens = None

                        assert confusion_rel.infons["type"] == "NONE"
                        if explanation_tokens:
                            confusion_explanations[(confusion_rel.infons["type"], rel_type)].append(explanation_tokens)

    return confusion_explanations, correct_explanations


if __name__ == '__main__':
    with open("saved_runs/explainable_bert/predictions.bioc.xml") as f:
        collection = bioc.load(f)
        confusion_explanations, correct_explanations = get_base_explanations(collection.documents)

    df = {"text": [], "label": [], "prob": [], "explanation": [], "cuid_head": [], "cuid_tail": []}
    with open("data/ctd/small_positive_explained.tsv") as f:
        for line in f:
            fields = line.strip().split("\t")
            if len(fields) < 2:
                continue

            text, label, prob, explanation, cuid_head, cuid_tail, _ = fields
            df["text"].append(text)
            df["label"].append(label)
            df["prob"].append(float(prob))
            df["explanation"].append(explanation)
            df["cuid_head"].append(cuid_head)
            df["cuid_tail"].append(cuid_tail)
    df = pd.DataFrame(df)
    df["head_type"] = "Chemical"
    df["tail_type"] = "Gene"
    df["pmid"] = 1
    df[["head_type", "cuid_head", "tail_type", "cuid_tail", "label", "text", "pmid"]].to_csv("data/ctd_relabeled/small_relabeled.tsv", sep="\t", header=None, index=False)

    df_global_confident = df[df.prob > df.prob.quantile(0.75)]
    df_global_confident[["head_type", "cuid_head", "tail_type", "cuid_tail", "label", "text", "pmid"]].to_csv("data/ctd_relabeled/small_relabeled_global_confident.tsv", sep="\t", header=None, index=False)

    df_local_confident = pd.DataFrame()
    for label in df.label.unique():
        df_label = df[df.label == label]
        df_local_confident = pd.concat([df_local_confident, df_label[df_label.prob > df_label.prob.quantile(0.75)]])
    df_local_confident[["head_type", "cuid_head", "tail_type", "cuid_tail", "label", "text", "pmid"]].to_csv("data/ctd_relabeled/small_relabeled_local_confident.tsv", sep="\t", header=None, index=False)

    convincing_explanation = []
    for _, row in df.iterrows():
        best_jaccard = 0.0
        best_match = None
        explanation_tokens = set(row.explanation.split(" "))
        all_correct_explanations = correct_explanations[row.label]
        for expl in all_correct_explanations:
            jaccard = len(expl & explanation_tokens) / len(expl | explanation_tokens)
            if jaccard > best_jaccard:
                best_match = expl & explanation_tokens
                best_jaccard = jaccard
        convincing_explanation.append(best_jaccard >= 0.34)
    df["convincing_explanation"] = convincing_explanation
    df_convincing_explanation = df[df["convincing_explanation"]]
    df_convincing_explanation[["head_type", "cuid_head", "tail_type", "cuid_tail", "label", "text", "pmid"]].to_csv("data/ctd_relabeled/small_relabeled_convincing_explanation.tsv", sep="\t", header=None, index=False)


