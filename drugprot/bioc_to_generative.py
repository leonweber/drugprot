from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path

import bioc
import numpy as np
import pandas as pd
from tqdm import tqdm

from drugprot.models.entity_marker_baseline import insert_consistently

LABEL_TO_SUBJECT_TEMPLATES = {
    "ACTIVATOR": "What gene does %s activate?",
    "AGONIST": "For which receptors is %s an agonist?",
    "AGONIST-ACTIVATOR": "For which receptors is %s an agonist that increases its response?",
    "AGONIST-INHIBITOR": "For which receptors is %s an agonist that decreases its response?",
    "ANTAGONIST": "For which genes is %s an antagonist?",
    "DIRECT-REGULATOR": "Which genes does %s regulate directly?",
    "INDIRECT-DOWNREGULATOR": "Which genes does %s downregulate indirectly?",
    "INDIRECT-UPREGULATOR": "Which genes does %s upregulate indirectly?",
    "INHIBITOR": "Which genes does %s inhibit?",
    "PART-OF": "Of which genes is %s a part?",
    "PRODUCT-OF": "Of which genes is %s a product",
    "SUBSTRATE": "For which proteins is %s a substrate?",
    "SUBSTRATE_PRODUCT-OF": "For which proteins is %s a substrate and a product at the same time?",
}
LABEL_TO_OBJECT_TEMPLATES = {
    "ACTIVATOR": "Which chemical activates %s?",
    "AGONIST": "Which chemicals are the agonists of %s?",
    "AGONIST-ACTIVATOR": "Which chemicals are the agonists of %s that increase its response?",
    "AGONIST-INHIBITOR": "Which chemicals are the agonists of %s an decreases its response?",
    "ANTAGONIST": "Which chemicals are the antagonists of %s?",
    "DIRECT-REGULATOR": "Which chemicals regulate %s directly?",
    "INDIRECT-DOWNREGULATOR": "Which chemicals downregulate %s indirectly?",
    "INDIRECT-UPREGULATOR": "Which chemicals upregulate %s indirectly?",
    "INHIBITOR": "Which chemicals inhibit %s?",
    "PART-OF": "Which chemicals are a part of %s?",
    "PRODUCT-OF": "Which chemicals are a product of %s?",
    "SUBSTRATE": "Which chemicals are a substrate of %s?",
    "SUBSTRATE_PRODUCT-OF": "Which chemicals are a substrate and a product of %s at the same time?",
}

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--doc-context", action="store_true")
    args = parser.parse_args()

    examples = {"source": [], "target": []}
    with args.input.open() as f:
        collection = bioc.load(f)
    for doc in tqdm(collection.documents):
        for sentence in doc.passages[0].sentences:
            ann_id_to_ann = {}
            for ann in sentence.annotations:
                ann_id_to_ann[ann.id] = ann

            head_rel_pair_to_tails = defaultdict(set)
            tail_rel_pair_to_heads = defaultdict(set)
            for rel in sentence.relations:
                head = rel.get_node("head").refid
                tail = rel.get_node("tail").refid
                assert (
                    ann_id_to_ann[head].infons["type"] == "CHEMICAL"
                    and "GENE" in ann_id_to_ann[tail].infons["type"]
                )
                rel = rel.infons["type"]
                head_rel_pair_to_tails[(head, rel)].add(tail)
                tail_rel_pair_to_heads[(tail, rel)].add(head)

            for head in sentence.annotations:
                if head.infons["type"] != "CHEMICAL":
                    continue
                for rel, template in LABEL_TO_SUBJECT_TEMPLATES.items():
                    template_filled = template % head.text
                    if args.doc_context:
                        context = doc.passages[0].text
                    else:
                        context = sentence.text

                    marked_context = context
                    starts = [head.locations[0].offset]
                    ends = [
                        head.locations[0].offset
                        + head.locations[0].length
                    ]
                    for tail in head_rel_pair_to_tails[(head.id, rel)]:
                        tail = ann_id_to_ann[tail]
                        starts.append(tail.locations[0].offset)
                        ends.append(
                            tail.locations[0].offset
                            + tail.locations[0].length
                        )

                    starts = np.array(starts)
                    ends = np.array(ends)
                    if not args.doc_context:
                        starts -= sentence.offset
                        ends -= sentence.offset

                    marked_context, starts, ends = insert_consistently(
                        offset=starts[0],
                        insertion="chemical* ",
                        starts=starts,
                        ends=ends,
                        text=marked_context,
                    )
                    marked_context, starts, ends = insert_consistently(
                        offset=ends[0],
                        insertion=" *chemical",
                        starts=starts,
                        ends=ends,
                        text=marked_context,
                    )
                    for i in range(len(starts[1:])):
                        marked_context, starts, ends = insert_consistently(
                            offset=starts[i+1],
                            insertion="gene* ",
                            starts=starts,
                            ends=ends,
                            text=marked_context,
                        )
                        marked_context, starts, ends = insert_consistently(
                            offset=ends[i+1],
                            insertion=" *gene",
                            starts=starts,
                            ends=ends,
                            text=marked_context,
                        )
                    examples["source"].append(f"question: {template_filled}\tcontext: {context}")
                    examples["target"].append(marked_context)

            for tail in sentence.annotations:
                if "GENE" not in tail.infons["type"]:
                    continue
                for rel, template in LABEL_TO_OBJECT_TEMPLATES.items():
                    template_filled = template % tail.text
                    if args.doc_context:
                        context = doc.passages[0].text
                    else:
                        context = sentence.text

                    marked_context = context
                    starts = [tail.locations[0].offset]
                    ends = [
                        tail.locations[0].offset
                        + tail.locations[0].length
                    ]
                    for head in tail_rel_pair_to_heads[(tail.id, rel)]:
                        head = ann_id_to_ann[head]
                        starts.append(head.locations[0].offset)
                        ends.append(
                            head.locations[0].offset
                            + head.locations[0].length
                        )

                    starts = np.array(starts)
                    ends = np.array(ends)
                    if not args.doc_context:
                        starts -= sentence.offset
                        ends -= sentence.offset

                    marked_context, starts, ends = insert_consistently(
                        offset=starts[0],
                        insertion="chemical* ",
                        starts=starts,
                        ends=ends,
                        text=marked_context,
                    )
                    marked_context, starts, ends = insert_consistently(
                        offset=ends[0],
                        insertion=" *chemical",
                        starts=starts,
                        ends=ends,
                        text=marked_context,
                    )
                    for i in range(len(starts[1:])):
                        marked_context, starts, ends = insert_consistently(
                            offset=starts[i+1],
                            insertion="gene* ",
                            starts=starts,
                            ends=ends,
                            text=marked_context,
                        )
                        marked_context, starts, ends = insert_consistently(
                            offset=ends[i+1],
                            insertion=" *gene",
                            starts=starts,
                            ends=ends,
                            text=marked_context,
                        )
                    examples["source"].append(f"question: {template_filled}\tcontext: {context}")
                    examples["target"].append(marked_context)


    df = pd.DataFrame(examples)
    df.to_csv(args.output)

