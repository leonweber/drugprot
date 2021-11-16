from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path

import bioc

from drugprot.models.entity_marker_baseline import BiocDataset, insert_pair_markers


def sentence_to_examples(sentence, doc_id):
    examples = []

    ann_id_to_ann = {}
    for ann in sentence.annotations:
        ann_id_to_ann[ann.id] = ann

    pair_to_relations = defaultdict(set)
    for rel in sentence.relations:
        head = rel.get_node("head").refid
        tail = rel.get_node("tail").refid
        rel = rel.infons["type"]
        pair_to_relations[(head, tail)].add(rel)

    for head in sentence.annotations:
        for tail in sentence.annotations:
            if not (head.infons["type"] == "CHEMICAL" and "GENE" in tail.infons["type"]):
                continue
            text = insert_pair_markers(
                text=sentence.text,
                head=head,
                tail=tail,
                sentence_offset=sentence.offset,
                mark_with_special_tokens=False, blind_entities=True
            )
            text = text.replace("@@HEAD-TAIL$$", "@CHEM-GENE$")
            text = text.replace("@HEAD$", "@CHEMICAL$")
            text = text.replace("@TAIL$", "@GENE$")
            labels = pair_to_relations[(head.id, tail.id)]
            if labels:
                label = sorted(labels)[0]
            else:
                label = "false"

            index = f"{doc_id}.{head.id}.{tail.id}"
            examples.append(
                f"{index}\t{text}\t{label}"
            )

    return examples



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    examples = []
    with args.input.open() as f:
        collection = bioc.load(f)
        for doc in collection.documents:
            for sentence in doc.passages[0].sentences:
                examples.extend(sentence_to_examples(sentence, doc.id))

    args.output: Path
    args.output.parent.mkdir(exist_ok=True)
    with args.output.open("w") as f:
        f.write("index\tsentence\tlabel\n")
        f.write("\n".join(examples))
