import argparse
from collections import defaultdict, Counter
from pathlib import Path

import bioc
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--drugprot-normalized", required=True, type=Path)

    args = parser.parse_args()

    with open(args.drugprot_normalized) as f:
        collection = bioc.load(f)

    pair_to_drugprot_relations = defaultdict(set)
    for doc in collection.documents:
        id_to_ann = {}
        for sentence in doc.passages[0].sentences:
            for ann in sentence.annotations:
                id_to_ann[ann.id] = ann

            for rel in sentence.relations:
                head = id_to_ann[rel.get_node("head").refid].infons["identifier"]
                tail = id_to_ann[rel.get_node("tail").refid].infons["identifier"]
                label = rel.infons["type"]
                if head.startswith("MESH") and tail.isnumeric():
                    pair_to_drugprot_relations[(head,tail)].add(label)


    pair_to_ctd_relations = defaultdict(set)
    df = pd.read_csv(args.input, sep="\t")
    for head, tail, rels in zip("MESH:" + df["ChemicalID"], df["GeneID"].astype(str), df["InteractionActions"].str.split("|")):
        pair_to_ctd_relations[(head, tail)].update(rels)

    overlap = set(pair_to_ctd_relations) & set(pair_to_drugprot_relations)
    print(f"Found {len(overlap)}/{len(pair_to_drugprot_relations)} DrugProt pairs in CTD")

    drugprot_to_ctd_mapping_count = defaultdict(list)
    for pair in overlap:
        for drugprot_rel in pair_to_drugprot_relations[pair]:
            drugprot_to_ctd_mapping_count[drugprot_rel].extend(pair_to_ctd_relations[pair])

    for drugprot_rel, ctd_rels in drugprot_to_ctd_mapping_count.items():
        print(drugprot_rel)
        print("=" * 8)
        print(Counter(ctd_rels).most_common())
        print()



